# Quantum-9-Bernstein_Vazirani



"""
Quantum_Bernstein_Vazirani.py

Python 3.11
Qiskit 1.4.4

Izvorni BV deo:
    H -> linearni orakl -> H
    Pomocni kubit je u stanju |->.
    Bez dodatne faze izlaz je tacno poznati niz s.

Prediktivna dopuna:
    s je prethodni broj iste sortirane pozicije, kodiran sa broj - 1.
    Dodatna faza orakla uci se iz svih dostupnih istorijskih prelaza.
    Regresija Bornove raspodele koristi Brier gubitak.
    Izlaz nije zaokruzena regresiona srednja vrednost.

Za jedan registar:
    delta(x) = sum_m theta[m] * (-1)**popcount(m & x)
    psi(y | s) =
        (1/64) sum_x (-1)**popcount((y xor s) & x) * exp(i*delta(x))

Dodatna faza je nasa adaptacija.
Opsti fazni orakl nema Boolean obecanje izvornog BV problema.
Izvorni BV slucaj proverava se zasebno sa theta = 0.

Kvantno:
    priprema, orakl, interferencija i Bornove verovatnoce.

Klasicno:
    optimizacija parametara, hronoloski izbor modela
    i tacno uslovljavanje na sedam rastucih razlicitih brojeva.

Tok optimizacije koristi tacan matricni simulator istog kola.
Qiskit Statevector proverava amplitude i daje zavrsne raspodele.
Nema slucajnog uzorkovanja kombinacija.
"""

import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_2980.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS

ORACLE_DEGREES = (2, 3)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

MAX_ITERATIONS = 300
GRADIENT_TOLERANCE = 1e-6
RESTARTS = 2

INDICES = np.arange(DIM, dtype=np.int64)
PARITY = np.array(
    [value.bit_count() & 1 for value in range(DIM)],
    dtype=np.int64,
)

HADAMARD = (
    1.0 - 2.0 * PARITY[
        np.bitwise_and(INDICES[:, None], INDICES[None, :])
    ]
) / np.sqrt(DIM)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    reader = csv.reader(raw.decode("utf-8-sig").splitlines())
    rows = []

    for line_number, row in enumerate(reader, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            all(1 <= value <= N for value in values),
            f"{path.name}, red {line_number}: broj van opsega 1..39.",
        )
        require(
            len(set(values)) == K,
            f"{path.name}, red {line_number}: ponovljen broj.",
        )
        require(
            values == sorted(values),
            f"{path.name}, red {line_number}: brojevi nisu rastuci.",
        )
        rows.append(values)

    require(
        len(rows) >= 5,
        f"{path.name}: nedovoljno redova za hronolosku podelu.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def make_basis(degree):
    require(
        isinstance(degree, int) and 1 <= degree <= QUBITS,
        "Neispravan stepen faznog orakla.",
    )

    masks = np.array(
        [
            value
            for value in range(1, DIM)
            if value.bit_count() <= degree
        ],
        dtype=np.int64,
    )

    basis = 1.0 - 2.0 * PARITY[
        np.bitwise_and(INDICES[:, None], masks[None, :])
    ]

    require(
        np.allclose(
            basis.T @ basis,
            DIM * np.eye(len(masks)),
            atol=1e-12,
            rtol=0.0,
        ),
        "Fazna baza nije ortogonalna.",
    )

    return masks, basis


def base_state(theta, basis):
    theta = np.asarray(theta, dtype=np.float64)

    require(
        theta.shape == (basis.shape[1],),
        "Neispravan oblik parametara orakla.",
    )
    require(
        np.isfinite(theta).all(),
        "Parametri orakla nisu konacni.",
    )

    phase = basis @ theta
    phase_state = np.exp(1j * phase) / np.sqrt(DIM)
    amplitudes = HADAMARD @ phase_state

    require(
        np.isfinite(amplitudes).all(),
        "Nekonacne amplitude.",
    )
    require(
        np.isclose(
            np.vdot(amplitudes, amplitudes).real,
            1.0,
            atol=2e-11,
            rtol=0.0,
        ),
        "Stanje nije normalizovano.",
    )

    return amplitudes, phase_state


def conditional_state(theta, basis, secret):
    require(0 <= secret < DIM, "BV ulaz je van opsega.")

    amplitudes, _ = base_state(theta, basis)

    return amplitudes[np.bitwise_xor(INDICES, secret)]


def amplitude_logs(amplitudes):
    absolute = np.abs(amplitudes)
    logs = np.full(absolute.shape, -np.inf, dtype=np.float64)
    positive = absolute > 0.0
    logs[positive] = 2.0 * np.log(absolute[positive])
    return logs


def generalized_oracle(phase):
    """
    U_delta = exp(-i * delta(x) * X_aux).

    Pomocni kubit |-> ima X sopstvenu vrednost -1.
    Zato ulaz |x>|-> dobija fazu exp(i*delta(x)).

    Pomocni kubit je najvisi bit Qiskit baznog indeksa.
    """
    phase = np.asarray(phase, dtype=np.float64)
    require(phase.shape == (DIM,), "Neispravan oblik faze.")

    cosine = np.diag(np.cos(phase))
    sine = np.diag(-1j * np.sin(phase))

    return np.block([
        [cosine, sine],
        [sine, cosine],
    ])


def qiskit_state(theta, basis, secret):
    require(0 <= secret < DIM, "Neispravan BV niz.")

    circuit = QuantumCircuit(QUBITS + 1)

    circuit.x(QUBITS)
    circuit.h(QUBITS)

    for qubit in range(QUBITS):
        circuit.h(qubit)

    # Originalni linearni BV orakl.
    for qubit in range(QUBITS):
        if (secret >> qubit) & 1:
            circuit.cx(qubit, QUBITS)

    # Nasa naucena fazna dopuna.
    if np.any(theta != 0.0):
        circuit.unitary(
            generalized_oracle(basis @ theta),
            list(range(QUBITS + 1)),
            label="Learned_phase_oracle",
        )

    for qubit in range(QUBITS):
        circuit.h(qubit)

    full_state = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=np.complex128,
    )

    expected_data = conditional_state(theta, basis, secret)
    expected_full = np.concatenate(
        [expected_data, -expected_data]
    ) / np.sqrt(2.0)

    require(
        np.allclose(
            full_state,
            expected_full,
            atol=2e-10,
            rtol=2e-10,
        ),
        "Qiskit i matricni simulator daju razlicite amplitude.",
    )

    # Uslovna amplituda uz poznatu faktorizaciju sa |->.
    data_state = np.sqrt(2.0) * full_state[:DIM]

    require(
        np.isclose(
            np.vdot(data_state, data_state).real,
            1.0,
            atol=2e-10,
            rtol=0.0,
        ),
        "Qiskit izlazni registar nije normalizovan.",
    )

    return data_state


def transition_counts(data, end):
    """
    Koriste se prelazi sa ciljnim redovima 1..end, ukljucivo.

    Zbog BV translacije:
        p(y | s) = p_base(y xor s).

    Brojanje XOR prelaza je tacna agregacija Brier gubitka.
    To nisu ucestalosti pojedinacnih loto brojeva.
    """
    require(
        1 <= end < len(data),
        "Neispravna granica istorijskih prelaza.",
    )

    transitions = np.bitwise_xor(
        data[1:end + 1],
        data[:end],
    )

    counts = np.zeros((K, DIM), dtype=np.float64)

    for slot in range(K):
        np.add.at(counts[slot], transitions[:, slot], 1.0)

    require(
        np.all(counts.sum(axis=1) == end),
        "Nisu obuhvaceni svi istorijski prelazi.",
    )

    return counts


def objective(theta, counts, basis, regularization):
    amplitudes, phase_state = base_state(theta, basis)

    count_sum = float(np.sum(counts))
    require(count_sum > 0.0, "Prazan skup prelaza.")

    target_distribution = counts / count_sum
    probabilities = np.abs(amplitudes) ** 2

    # Srednji Brier gubitak prema pojedinacnim one-hot ciljevima.
    loss = (
        np.dot(probabilities, probabilities)
        - 2.0 * np.dot(target_distribution, probabilities)
        + 1.0
    )

    amplitude_adjoint = (
        4.0
        * (probabilities - target_distribution)
        * amplitudes
    )
    phase_adjoint = HADAMARD.T @ amplitude_adjoint

    gradient = basis.T @ np.real(
        np.conjugate(phase_adjoint)
        * (1j * phase_state)
    )

    loss += regularization * np.dot(theta, theta)
    gradient += 2.0 * regularization * theta

    require(
        np.isfinite(loss) and np.isfinite(gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), np.asarray(gradient, dtype=np.float64)


def initial_parameters(size, restart):
    indices = np.arange(1, size + 1, dtype=np.float64)

    if restart == 0:
        return 0.22 * np.sin(indices)

    return 0.22 * np.cos(indices * np.sqrt(2.0))


def fit_models(counts, basis, regularization, label):
    models = []

    for slot in range(K):
        best = {
            "loss": np.inf,
            "theta": None,
            "success": False,
            "message": "",
        }

        for restart in range(RESTARTS):
            tracked = {
                "loss": np.inf,
                "theta": None,
            }

            def tracked_objective(theta):
                value, gradient = objective(
                    theta,
                    counts[slot],
                    basis,
                    regularization,
                )

                if value < tracked["loss"]:
                    tracked["loss"] = value
                    tracked["theta"] = theta.copy()

                return value, gradient

            result = minimize(
                tracked_objective,
                initial_parameters(basis.shape[1], restart),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": MAX_ITERATIONS,
                    "gtol": GRADIENT_TOLERANCE,
                    "ftol": 1e-12,
                    "maxls": 40,
                },
            )

            require(
                tracked["theta"] is not None,
                "Optimizator nije dao konacan kandidat.",
            )

            if tracked["loss"] < best["loss"]:
                best = {
                    "loss": tracked["loss"],
                    "theta": tracked["theta"].copy(),
                    "success": bool(result.success),
                    "message": str(result.message),
                }

        theta = best["theta"]
        value, gradient = objective(
            theta,
            counts[slot],
            basis,
            regularization,
        )
        gradient_norm = float(np.max(np.abs(gradient)))

        print(
            f"{label}, registar {slot + 1}, "
            f"Brier+L2={value:.9f}, "
            f"max|grad|={gradient_norm:.3e}"
        )

        if (
            not best["success"]
            or gradient_norm > 10.0 * GRADIENT_TOLERANCE
        ):
            print("Napomena: optimizacija nije potvrdila mali gradijent.")
            print(best["message"])

        models.append(theta.copy())

    return models


def make_log_kernels(models, basis):
    """
    kernel[slot][y, s] = log p(y | s).

    Sva 64 stanja ostaju u normalizovanoj kvantnoj raspodeli.
    Uslovljavanje na Loto 1..39 sledi tek u joint_distribution.
    """
    kernels = []

    for theta in models:
        amplitudes, _ = base_state(theta, basis)
        base_logs = amplitude_logs(amplitudes)

        kernel = base_logs[
            np.bitwise_xor(
                INDICES[:, None],
                np.arange(N, dtype=np.int64)[None, :],
            )
        ]

        require(
            np.allclose(
                logsumexp(kernel, axis=0),
                0.0,
                atol=2e-10,
                rtol=0.0,
            ),
            "Uslovna kvantna raspodela nije normalizovana.",
        )

        kernels.append(kernel)

    return kernels


def joint_distribution(log_weights):
    """
    Tacna raspodela proizvoda registara uslovljena na:
        0 <= y0 < y1 < ... < y6 < 39.

    Dinamicko programiranje daje:
        log normalizator,
        marginale,
        globalni MAP.

    Nema popravke duplikata i nema kozmetickih filtera.
    """
    logs = np.asarray(log_weights, dtype=np.float64).copy()

    require(logs.ndim == 2, "Potreban je dvodimenzioni niz.")
    require(
        not np.isnan(logs).any() and not np.isposinf(logs).any(),
        "Neispravne log verovatnoce.",
    )

    k, n = logs.shape
    require(1 <= k <= n, "Neispravna dimenzija kombinacije.")

    for slot in range(k):
        logs[slot, :slot] = -np.inf
        logs[slot, n - k + slot + 1:] = -np.inf

    forward = np.full((k, n), -np.inf)
    maximum = np.full((k, n), -np.inf)
    parent = np.full((k, n), -1, dtype=np.int64)

    forward[0] = logs[0]
    maximum[0] = logs[0]

    for slot in range(1, k):
        for value in range(slot, n - k + slot + 1):
            forward[slot, value] = (
                logs[slot, value]
                + logsumexp(forward[slot - 1, :value])
            )

            previous = int(np.argmax(maximum[slot - 1, :value]))
            maximum[slot, value] = (
                logs[slot, value] + maximum[slot - 1, previous]
            )
            parent[slot, value] = previous

    log_z = float(logsumexp(forward[-1]))

    if not np.isfinite(log_z):
        raise DistributionError(
            "Model nema pozitivnu masu na legalnim kombinacijama."
        )

    backward = np.full((k, n), -np.inf)
    backward[-1] = 0.0

    for slot in range(k - 2, -1, -1):
        for value in range(slot, n - k + slot + 1):
            backward[slot, value] = logsumexp(
                logs[slot + 1, value + 1:]
                + backward[slot + 1, value + 1:]
            )

    marginals = np.exp(forward + backward - log_z)

    path = np.empty(k, dtype=np.int64)
    path[-1] = int(np.argmax(maximum[-1]))

    for slot in range(k - 1, 0, -1):
        path[slot - 1] = parent[slot, path[slot]]

    log_mode = float(logs[np.arange(k), path].sum())

    require(
        np.all(np.diff(path) > 0),
        "MAP kombinacija nije strogo rastuca.",
    )
    require(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=2e-9,
            rtol=0.0,
        ),
        "Uslovne marginale nisu normalizovane.",
    )
    require(
        log_mode <= log_z + 2e-9,
        "MAP masa je veca od ukupne legalne mase.",
    )
    require(
        log_z <= 2e-9,
        "Legalna masa je veca od jedan.",
    )

    return {
        "path": path,
        "marginals": marginals,
        "log_z": log_z,
        "log_mode_probability": log_mode - log_z,
        "logs": logs,
    }


def weights_for_previous(kernels, previous):
    return np.stack(
        [
            kernels[slot][:N, int(previous[slot])]
            for slot in range(K)
        ],
        axis=0,
    )


def evaluate(models, basis, data, times):
    kernels = make_log_kernels(models, basis)
    losses = []
    hits = []
    zero_targets = 0

    for time in times:
        weights = weights_for_previous(kernels, data[time - 1])
        joint = joint_distribution(weights)

        target_log_mass = float(
            joint["logs"][np.arange(K), data[time]].sum()
        )

        if np.isneginf(target_log_mass):
            nll = np.inf
            zero_targets += 1
        else:
            nll = joint["log_z"] - target_log_mass
            require(
                np.isfinite(nll) and nll >= -2e-8,
                "Neispravan uslovni NLL.",
            )
            nll = max(0.0, nll)

        overlap = len(
            set(joint["path"].tolist())
            & set(data[time].tolist())
        )

        losses.append(nll)
        hits.append(overlap)

    require(len(losses) > 0, "Prazan skup za evaluaciju.")

    return {
        "nll": float(np.mean(losses)),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": zero_targets,
    }


def print_metrics(label, metrics):
    print(label)
    print(f"Srednji uslovni NLL: {metrics['nll']:.9f}")
    print(f"Srednji broj pogodaka: {metrics['hits'].mean():.6f}")
    print(f"Ciljevi sa nultom masom: {metrics['zero_targets']}")

    histogram = np.bincount(metrics["hits"], minlength=K + 1)

    for count in range(K + 1):
        print(f"{count} pogodaka: {int(histogram[count])}")


def self_checks():
    require(
        np.allclose(
            HADAMARD.T @ HADAMARD,
            np.eye(DIM),
            atol=1e-12,
            rtol=0.0,
        ),
        "Hadamard transformacija nije unitarna.",
    )

    _, basis = make_basis(2)
    zero = np.zeros(basis.shape[1])

    # Originalni BV slucaj za svih 64 bitna niza.
    for secret in range(DIM):
        amplitudes = qiskit_state(zero, basis, secret)
        expected = np.zeros(DIM, dtype=np.complex128)
        expected[secret] = 1.0

        require(
            np.allclose(
                amplitudes,
                expected,
                atol=2e-10,
                rtol=0.0,
            ),
            f"Izvorni BV nije vratio niz {secret:06b}.",
        )

    for degree in ORACLE_DEGREES:
        _, current_basis = make_basis(degree)
        theta = initial_parameters(current_basis.shape[1], 0)

        counts = (
            1.0
            + np.sin(np.arange(DIM, dtype=np.float64) + 0.3)
        ) ** 2

        _, analytic = objective(
            theta,
            counts,
            current_basis,
            0.003,
        )
        numeric = np.empty_like(theta)
        step = 1e-6

        for index in range(len(theta)):
            upper = theta.copy()
            lower = theta.copy()
            upper[index] += step
            lower[index] -= step

            numeric[index] = (
                objective(
                    upper, counts, current_basis, 0.003
                )[0]
                - objective(
                    lower, counts, current_basis, 0.003
                )[0]
            ) / (2.0 * step)

        require(
            np.allclose(
                analytic,
                numeric,
                atol=2e-6,
                rtol=2e-4,
            ),
            "Analiticki gradijent ne odgovara konacnoj razlici.",
        )

        for secret in (0, 1, 17, 38, 63):
            qiskit_state(theta, current_basis, secret)

    # Provera globalnog MAP-a i normalizatora.
    toy = np.exp(
        0.5 * np.sin(np.arange(15, dtype=np.float64).reshape(3, 5))
    )
    toy /= toy.sum(axis=1, keepdims=True)
    toy_logs = np.log(toy)

    result = joint_distribution(toy_logs)

    combinations = list(itertools.combinations(range(5), 3))
    scores = np.array(
        [
            toy_logs[np.arange(3), combination].sum()
            for combination in combinations
        ],
        dtype=np.float64,
    )

    require(
        np.isclose(
            result["log_z"],
            logsumexp(scores),
            atol=1e-12,
            rtol=0.0,
        ),
        "DP normalizator ne odgovara potpunom pregledu.",
    )
    require(
        np.isclose(
            toy_logs[np.arange(3), result["path"]].sum(),
            scores.max(),
            atol=1e-12,
            rtol=0.0,
        ),
        "DP nije dao globalni maksimum.",
    )

    brute_marginals = np.zeros_like(toy)
    masses = np.exp(scores - logsumexp(scores))

    for combination, mass in zip(combinations, masses):
        brute_marginals[np.arange(3), combination] += mass

    require(
        np.allclose(
            result["marginals"],
            brute_marginals,
            atol=1e-12,
            rtol=0.0,
        ),
        "DP marginale ne odgovaraju potpunom pregledu.",
    )

    # Jedina moguca kombinacija.
    sole = np.full((3, 5), -np.inf)
    sole[np.arange(3), [0, 2, 4]] = 0.0
    sole_result = joint_distribution(sole)

    require(
        np.array_equal(sole_result["path"], [0, 2, 4]),
        "Greska pri jedinoj legalnoj kombinaciji.",
    )
    require(
        np.isclose(sole_result["log_z"], 0.0),
        "Greska normalizatora za jedinu kombinaciju.",
    )

    # Nepostojeca legalna podrska mora biti prijavljena.
    impossible = np.full((3, 5), -np.inf)
    impossible[:, 0] = 0.0

    try:
        joint_distribution(impossible)
    except DistributionError:
        pass
    else:
        raise RuntimeError(
            "Nulta legalna masa nije prijavljena."
        )

    # Racunanje log amplitude cuva vrlo male vrednosti.
    tiny_log = amplitude_logs(
        np.array([1e-180 + 0j], dtype=np.complex128)
    )[0]
    require(
        np.isfinite(tiny_log),
        "Log amplituda je nepotrebno izgubio malu vrednost.",
    )

    print("Ugradjene provere BV kola, gradijenta i DP-a su prosle.")


def longest_consecutive(values):
    best = 1
    current = 1

    for left, right in zip(values[:-1], values[1:]):
        if right == left + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1

    return best


def process_dataset(dataset):
    path = dataset["path"]
    data = dataset["data"]

    transitions = len(data) - 1
    train_end = int(TRAIN_FRACTION * transitions)
    validation_end = int(
        VALIDATION_END_FRACTION * transitions
    )

    require(
        0 < train_end < validation_end < transitions,
        f"{path.name}: nedovoljno podataka za tri hronoloska dela.",
    )

    print()
    print("=" * 72)
    print(path.name)
    print(f"Stvarni broj kola: {len(data)}")
    print(f"SHA256: {dataset['sha256']}")
    print(f"Trening prelazi: 1..{train_end}")
    print(
        f"Validacioni prelazi: "
        f"{train_end + 1}..{validation_end}"
    )
    print(
        f"Test prelazi: "
        f"{validation_end + 1}..{transitions}"
    )

    train_counts = transition_counts(data, train_end)
    best = None

    for degree in ORACLE_DEGREES:
        masks, basis = make_basis(degree)

        for regularization in REGULARIZATIONS:
            label = (
                f"VALIDACIJA degree={degree}, "
                f"L2={regularization}"
            )

            models = fit_models(
                train_counts,
                basis,
                regularization,
                label,
            )

            try:
                metrics = evaluate(
                    models,
                    basis,
                    data,
                    range(train_end + 1, validation_end + 1),
                )
            except DistributionError as exc:
                print(f"Kandidat odbijen: {exc}")
                continue

            print_metrics(label, metrics)

            if not np.isfinite(metrics["nll"]):
                print("Kandidat nema konacan validacioni NLL.")
                continue

            if best is None or metrics["nll"] < best["nll"]:
                best = {
                    "degree": degree,
                    "regularization": regularization,
                    "nll": metrics["nll"],
                    "parameter_count": len(masks),
                }

    require(
        best is not None,
        f"{path.name}: nema validnog kandidata.",
    )

    print()
    print(
        f"Izabrano: degree={best['degree']}, "
        f"faza-parametara={best['parameter_count']}, "
        f"L2={best['regularization']}"
    )

    _, basis = make_basis(best["degree"])

    # Ponovno ucenje na treningu i validaciji.
    # Test ciljevi jos nisu ukljuceni.
    test_models = fit_models(
        transition_counts(data, validation_end),
        basis,
        best["regularization"],
        "TEST TRENING",
    )

    test_metrics = evaluate(
        test_models,
        basis,
        data,
        range(validation_end + 1, len(data)),
    )
    print_metrics("ODVOJENI HRONOLOSKI TEST", test_metrics)

    # Zavrsno ucenje obuhvata celu raspolozivu istoriju.
    final_models = fit_models(
        transition_counts(data, transitions),
        basis,
        best["regularization"],
        "CELA ISTORIJA",
    )

    kernels = make_log_kernels(final_models, basis)
    previous = data[-1]
    numerical_logs = weights_for_previous(kernels, previous)

    # Zavrsna raspodela dobija se neposredno iz Qiskit amplituda.
    qiskit_logs = np.empty((K, N), dtype=np.float64)

    for slot in range(K):
        state = qiskit_state(
            final_models[slot],
            basis,
            int(previous[slot]),
        )
        qiskit_logs[slot] = amplitude_logs(state)[:N]

        require(
            np.allclose(
                np.abs(state) ** 2,
                np.exp(kernels[slot][:, int(previous[slot])]),
                atol=2e-10,
                rtol=2e-9,
            ),
            "Zavrsne Qiskit verovatnoce se razlikuju od modela.",
        )

    # Log vrednosti oko numerickih nula mogu se razlikovati.
    # Uporedjuju se njihove stvarne verovatnoce.
    require(
        np.allclose(
            np.exp(qiskit_logs),
            np.exp(numerical_logs),
            atol=2e-10,
            rtol=2e-9,
        ),
        "Zavrsna raspodela nije dosledna.",
    )

    joint = joint_distribution(qiskit_logs)
    combination = joint["path"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Zavrsna kombinacija nije legalna.",
    )

    print()
    print(f"Log legalne mase: {joint['log_z']:.9f}")
    print(
        f"Log uslovne mase MAP-a: "
        f"{joint['log_mode_probability']:.9f}"
    )

    # Dijagnostika osetljivosti na poznati BV ulaz.
    maximum_context_change = max(
        float(
            np.max(
                np.abs(
                    np.exp(kernel[:N, :])
                    - np.exp(kernel[:N, :1])
                )
            )
        )
        for kernel in kernels
    )

    print(
        f"Najveca promena raspodele kroz BV ulaze: "
        f"{maximum_context_change:.9e}"
    )

    if maximum_context_change < 1e-10:
        print(
            "Upozorenje: raspodela je gotovo neosetljiva "
            "na prethodni broj."
        )

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT je isti kao poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(
            f"Upozorenje: NEXT sadrzi niz od {run} "
            "uzastopnih brojeva."
        )

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    span_percentile = 100.0 * float(
        np.mean(historical_spans <= span)
    )

    print(f"Raspon NEXT kombinacije: {span}")
    print(
        f"Udeo istorijskih raspona <= NEXT raspon: "
        f"{span_percentile:.3f}%"
    )

    if span_percentile <= 1.0:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    return {
        "name": path.name,
        "sha256": dataset["sha256"],
        "next": combination,
        "raw": np.exp(qiskit_logs),
        "marginals": joint["marginals"],
    }


def main():
    print("Quantum Bernstein-Vazirani")
    print(f"Qiskit: {qiskit.__version__}")
    print("Sedam izlaznih registara sa po sest kubita.")
    print("Svako pojedinacno kolo koristi i jedan pomocni kubit.")
    print("Registri se simuliraju zasebno.")
    print("Originalni BV orakl koristi prethodni poznati broj.")
    print("Dodatna faza orakla uci se iz istorijskih XOR prelaza.")
    print("L-BFGS-B je klasicni optimizator kvantnih faza.")
    print("Izbor modela koristi samo hronolosku validaciju.")
    print("NEXT koristi celu istoriju svakog CSV-a zasebno.")
    print()

    self_checks()

    results = []

    for path in FILES:
        dataset = load_csv(path)
        results.append(process_dataset(dataset))

    print()
    print("=" * 72)
    print("POREDJENJE TRI NEZAVISNA REZULTATA")

    for left, right in itertools.combinations(results, 2):
        if left["sha256"] == right["sha256"]:
            print(
                f"Upozorenje: isti sadrzaj fajlova "
                f"{left['name']} i {right['name']}."
            )

        if np.array_equal(left["next"], right["next"]):
            print(
                f"Upozorenje: ista NEXT kombinacija za "
                f"{left['name']} i {right['name']}."
            )
            print(
                "Najveca razlika sirovih raspodela: "
                f"{np.max(np.abs(left['raw'] - right['raw'])):.9e}"
            )
            print(
                "Najveca razlika uslovnih marginala: "
                f"{np.max(np.abs(left['marginals'] - right['marginals'])):.9e}"
            )
            print(
                "Poklapanje zahteva pregled raspodela "
                "i rezultata testa."
            )

    print()
    print("=" * 72)

    for result in results:
        numbers = " ".join(
            str(int(value)) for value in result["next"]
        )
        print(f"{result['name']} NEXT: {numbers}")


if __name__ == "__main__":
    main()











"""
Quantum Bernstein-Vazirani
Qiskit: 1.4.4
Sedam izlaznih registara sa po sest kubita.
Svako pojedinacno kolo koristi i jedan pomocni kubit.
Registri se simuliraju zasebno.
Originalni BV orakl koristi prethodni poznati broj.
Dodatna faza orakla uci se iz istorijskih XOR prelaza.
L-BFGS-B je klasicni optimizator kvantnih faza.
Izbor modela koristi samo hronolosku validaciju.
NEXT koristi celu istoriju svakog CSV-a zasebno.

Ugradjene provere BV kola, gradijenta i DP-a su prosle.

========================================================================
loto7_4716_k89.csv
Stvarni broj kola: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelazi: 1..3300
Validacioni prelazi: 3301..4007
Test prelazi: 4008..4715
VALIDACIJA degree=2, L2=0.001, registar 1, Brier+L2=0.929911570, max|grad|=7.697e-07
VALIDACIJA degree=2, L2=0.001, registar 2, Brier+L2=0.960664377, max|grad|=8.603e-07
VALIDACIJA degree=2, L2=0.001, registar 3, Brier+L2=0.967926578, max|grad|=7.539e-07
VALIDACIJA degree=2, L2=0.001, registar 4, Brier+L2=0.971956842, max|grad|=5.351e-07
VALIDACIJA degree=2, L2=0.001, registar 5, Brier+L2=0.970080138, max|grad|=5.565e-07
VALIDACIJA degree=2, L2=0.001, registar 6, Brier+L2=0.965953617, max|grad|=7.924e-07
VALIDACIJA degree=2, L2=0.001, registar 7, Brier+L2=0.945193593, max|grad|=8.823e-07
VALIDACIJA degree=2, L2=0.001
Srednji uslovni NLL: 17.344718619
Srednji broj pogodaka: 1.229137
Ciljevi sa nultom masom: 0
0 pogodaka: 176
1 pogodaka: 272
2 pogodaka: 191
3 pogodaka: 57
4 pogodaka: 11
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=2, L2=0.01, registar 1, Brier+L2=0.947604361, max|grad|=5.458e-07
VALIDACIJA degree=2, L2=0.01, registar 2, Brier+L2=0.982231614, max|grad|=5.487e-07
VALIDACIJA degree=2, L2=0.01, registar 3, Brier+L2=0.990684147, max|grad|=9.757e-07
VALIDACIJA degree=2, L2=0.01, registar 4, Brier+L2=0.994151602, max|grad|=4.601e-07
VALIDACIJA degree=2, L2=0.01, registar 5, Brier+L2=0.992319528, max|grad|=9.515e-07
VALIDACIJA degree=2, L2=0.01, registar 6, Brier+L2=0.995094242, max|grad|=9.791e-07
VALIDACIJA degree=2, L2=0.01, registar 7, Brier+L2=0.967006201, max|grad|=5.670e-07
VALIDACIJA degree=2, L2=0.01
Srednji uslovni NLL: 18.383185541
Srednji broj pogodaka: 1.248939
Ciljevi sa nultom masom: 0
0 pogodaka: 168
1 pogodaka: 280
2 pogodaka: 188
3 pogodaka: 58
4 pogodaka: 12
5 pogodaka: 1
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.001, registar 1, Brier+L2=0.931566922, max|grad|=9.542e-07
VALIDACIJA degree=3, L2=0.001, registar 2, Brier+L2=0.960203479, max|grad|=4.836e-07
VALIDACIJA degree=3, L2=0.001, registar 3, Brier+L2=0.967263956, max|grad|=8.221e-07
VALIDACIJA degree=3, L2=0.001, registar 4, Brier+L2=0.970559969, max|grad|=6.116e-07
VALIDACIJA degree=3, L2=0.001, registar 5, Brier+L2=0.969285963, max|grad|=5.769e-07
VALIDACIJA degree=3, L2=0.001, registar 6, Brier+L2=0.967185360, max|grad|=7.334e-07
VALIDACIJA degree=3, L2=0.001, registar 7, Brier+L2=0.940893217, max|grad|=9.258e-07
VALIDACIJA degree=3, L2=0.001
Srednji uslovni NLL: 17.561380344
Srednji broj pogodaka: 1.248939
Ciljevi sa nultom masom: 0
0 pogodaka: 168
1 pogodaka: 280
2 pogodaka: 188
3 pogodaka: 58
4 pogodaka: 12
5 pogodaka: 1
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.01, registar 1, Brier+L2=0.945005100, max|grad|=7.362e-07
VALIDACIJA degree=3, L2=0.01, registar 2, Brier+L2=0.977173780, max|grad|=9.207e-07
VALIDACIJA degree=3, L2=0.01, registar 3, Brier+L2=0.985840597, max|grad|=7.493e-07
VALIDACIJA degree=3, L2=0.01, registar 4, Brier+L2=0.988728863, max|grad|=8.621e-07
VALIDACIJA degree=3, L2=0.01, registar 5, Brier+L2=0.987065383, max|grad|=7.089e-07
VALIDACIJA degree=3, L2=0.01, registar 6, Brier+L2=0.985468323, max|grad|=4.963e-07
VALIDACIJA degree=3, L2=0.01, registar 7, Brier+L2=0.958086594, max|grad|=5.938e-07
VALIDACIJA degree=3, L2=0.01
Srednji uslovni NLL: 17.920766186
Srednji broj pogodaka: 1.248939
Ciljevi sa nultom masom: 0
0 pogodaka: 168
1 pogodaka: 280
2 pogodaka: 188
3 pogodaka: 58
4 pogodaka: 12
5 pogodaka: 1
6 pogodaka: 0
7 pogodaka: 0

Izabrano: degree=2, faza-parametara=21, L2=0.001
TEST TRENING, registar 1, Brier+L2=0.930782216, max|grad|=6.147e-07
TEST TRENING, registar 2, Brier+L2=0.960674990, max|grad|=7.643e-07
TEST TRENING, registar 3, Brier+L2=0.968082218, max|grad|=4.946e-07
TEST TRENING, registar 4, Brier+L2=0.972026978, max|grad|=7.687e-07
TEST TRENING, registar 5, Brier+L2=0.970344642, max|grad|=9.578e-07
TEST TRENING, registar 6, Brier+L2=0.965592467, max|grad|=6.863e-07
TEST TRENING, registar 7, Brier+L2=0.946199078, max|grad|=4.475e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.559656729
Srednji broj pogodaka: 1.282486
Ciljevi sa nultom masom: 0
0 pogodaka: 136
1 pogodaka: 306
2 pogodaka: 210
3 pogodaka: 43
4 pogodaka: 12
5 pogodaka: 1
6 pogodaka: 0
7 pogodaka: 0
CELA ISTORIJA, registar 1, Brier+L2=0.930654671, max|grad|=7.182e-07
CELA ISTORIJA, registar 2, Brier+L2=0.960573096, max|grad|=9.799e-07
CELA ISTORIJA, registar 3, Brier+L2=0.968143819, max|grad|=3.161e-07
CELA ISTORIJA, registar 4, Brier+L2=0.971887976, max|grad|=7.609e-07
CELA ISTORIJA, registar 5, Brier+L2=0.970243048, max|grad|=5.642e-07
CELA ISTORIJA, registar 6, Brier+L2=0.965407613, max|grad|=8.199e-07
CELA ISTORIJA, registar 7, Brier+L2=0.945861355, max|grad|=7.832e-07

Log legalne mase: -4.946226000
Log uslovne mase MAP-a: -14.644973734
Najveca promena raspodele kroz BV ulaze: 1.005294158e-01
Raspon NEXT kombinacije: 28
Udeo istorijskih raspona <= NEXT raspon: 34.818%

========================================================================
loto7_4716_k89_loto_2980.csv
Stvarni broj kola: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelazi: 1..2084
Validacioni prelazi: 2085..2531
Test prelazi: 2532..2978
VALIDACIJA degree=2, L2=0.001, registar 1, Brier+L2=0.930666912, max|grad|=9.407e-07
VALIDACIJA degree=2, L2=0.001, registar 2, Brier+L2=0.959715273, max|grad|=8.984e-07
VALIDACIJA degree=2, L2=0.001, registar 3, Brier+L2=0.969057691, max|grad|=4.817e-07
VALIDACIJA degree=2, L2=0.001, registar 4, Brier+L2=0.972295087, max|grad|=9.394e-07
VALIDACIJA degree=2, L2=0.001, registar 5, Brier+L2=0.969636480, max|grad|=4.913e-07
VALIDACIJA degree=2, L2=0.001, registar 6, Brier+L2=0.965985756, max|grad|=6.942e-07
VALIDACIJA degree=2, L2=0.001, registar 7, Brier+L2=0.946166288, max|grad|=4.093e-07
VALIDACIJA degree=2, L2=0.001
Srednji uslovni NLL: 17.628310359
Srednji broj pogodaka: 1.228188
Ciljevi sa nultom masom: 0
0 pogodaka: 107
1 pogodaka: 178
2 pogodaka: 116
3 pogodaka: 45
4 pogodaka: 1
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=2, L2=0.01, registar 1, Brier+L2=0.948845736, max|grad|=9.307e-07
VALIDACIJA degree=2, L2=0.01, registar 2, Brier+L2=0.980765051, max|grad|=4.559e-07
VALIDACIJA degree=2, L2=0.01, registar 3, Brier+L2=0.992680626, max|grad|=8.455e-07
VALIDACIJA degree=2, L2=0.01, registar 4, Brier+L2=0.994484214, max|grad|=9.190e-07
VALIDACIJA degree=2, L2=0.01, registar 5, Brier+L2=0.991704845, max|grad|=9.253e-07
VALIDACIJA degree=2, L2=0.01, registar 6, Brier+L2=0.996339832, max|grad|=6.980e-07
VALIDACIJA degree=2, L2=0.01, registar 7, Brier+L2=0.965562453, max|grad|=8.007e-07
VALIDACIJA degree=2, L2=0.01
Srednji uslovni NLL: 18.563219895
Srednji broj pogodaka: 1.243848
Ciljevi sa nultom masom: 0
0 pogodaka: 97
1 pogodaka: 184
2 pogodaka: 129
3 pogodaka: 34
4 pogodaka: 3
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.001, registar 1, Brier+L2=0.930281479, max|grad|=9.991e-07
VALIDACIJA degree=3, L2=0.001, registar 2, Brier+L2=0.958988096, max|grad|=6.297e-07
VALIDACIJA degree=3, L2=0.001, registar 3, Brier+L2=0.967082625, max|grad|=9.928e-07
VALIDACIJA degree=3, L2=0.001, registar 4, Brier+L2=0.970950114, max|grad|=7.389e-07
VALIDACIJA degree=3, L2=0.001, registar 5, Brier+L2=0.969338125, max|grad|=8.117e-07
VALIDACIJA degree=3, L2=0.001, registar 6, Brier+L2=0.967051388, max|grad|=9.499e-07
VALIDACIJA degree=3, L2=0.001, registar 7, Brier+L2=0.939304851, max|grad|=8.789e-07
VALIDACIJA degree=3, L2=0.001
Srednji uslovni NLL: 17.537276490
Srednji broj pogodaka: 1.203579
Ciljevi sa nultom masom: 0
0 pogodaka: 110
1 pogodaka: 167
2 pogodaka: 140
3 pogodaka: 29
4 pogodaka: 1
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.01, registar 1, Brier+L2=0.945093878, max|grad|=6.482e-07
VALIDACIJA degree=3, L2=0.01, registar 2, Brier+L2=0.975767025, max|grad|=7.561e-07
VALIDACIJA degree=3, L2=0.01, registar 3, Brier+L2=0.986307932, max|grad|=5.674e-07
VALIDACIJA degree=3, L2=0.01, registar 4, Brier+L2=0.989207498, max|grad|=9.555e-07
VALIDACIJA degree=3, L2=0.01, registar 5, Brier+L2=0.986613521, max|grad|=9.313e-07
VALIDACIJA degree=3, L2=0.01, registar 6, Brier+L2=0.985088192, max|grad|=8.382e-07
VALIDACIJA degree=3, L2=0.01, registar 7, Brier+L2=0.958899863, max|grad|=6.659e-07
VALIDACIJA degree=3, L2=0.01
Srednji uslovni NLL: 17.928639293
Srednji broj pogodaka: 1.277405
Ciljevi sa nultom masom: 0
0 pogodaka: 93
1 pogodaka: 184
2 pogodaka: 128
3 pogodaka: 37
4 pogodaka: 5
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0

Izabrano: degree=3, faza-parametara=41, L2=0.001
TEST TRENING, registar 1, Brier+L2=0.931168627, max|grad|=7.163e-07
TEST TRENING, registar 2, Brier+L2=0.959899015, max|grad|=7.880e-07
TEST TRENING, registar 3, Brier+L2=0.967182637, max|grad|=6.945e-07
TEST TRENING, registar 4, Brier+L2=0.970657293, max|grad|=7.501e-07
TEST TRENING, registar 5, Brier+L2=0.969189572, max|grad|=6.552e-07
TEST TRENING, registar 6, Brier+L2=0.966729764, max|grad|=4.758e-07
TEST TRENING, registar 7, Brier+L2=0.937733531, max|grad|=7.675e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.647358859
Srednji broj pogodaka: 1.275168
Ciljevi sa nultom masom: 0
0 pogodaka: 93
1 pogodaka: 179
2 pogodaka: 138
3 pogodaka: 33
4 pogodaka: 4
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
CELA ISTORIJA, registar 1, Brier+L2=0.931506580, max|grad|=6.949e-07
CELA ISTORIJA, registar 2, Brier+L2=0.959771167, max|grad|=7.925e-07
CELA ISTORIJA, registar 3, Brier+L2=0.967468080, max|grad|=7.600e-07
CELA ISTORIJA, registar 4, Brier+L2=0.970826807, max|grad|=9.302e-07
CELA ISTORIJA, registar 5, Brier+L2=0.969188871, max|grad|=5.032e-07
CELA ISTORIJA, registar 6, Brier+L2=0.966222033, max|grad|=4.457e-07
CELA ISTORIJA, registar 7, Brier+L2=0.937620872, max|grad|=7.521e-07

Log legalne mase: -5.357294326
Log uslovne mase MAP-a: -13.849912198
Najveca promena raspodele kroz BV ulaze: 1.092595148e-01
Raspon NEXT kombinacije: 22
Udeo istorijskih raspona <= NEXT raspon: 9.869%

========================================================================
loto7_4716_k89_loto_plus_1736.csv
Stvarni broj kola: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelazi: 1..1213
Validacioni prelazi: 1214..1473
Test prelazi: 1474..1734
VALIDACIJA degree=2, L2=0.001, registar 1, Brier+L2=0.926289755, max|grad|=7.259e-07
VALIDACIJA degree=2, L2=0.001, registar 2, Brier+L2=0.959775708, max|grad|=7.510e-07
VALIDACIJA degree=2, L2=0.001, registar 3, Brier+L2=0.970083012, max|grad|=6.060e-07
VALIDACIJA degree=2, L2=0.001, registar 4, Brier+L2=0.971521053, max|grad|=9.462e-07
VALIDACIJA degree=2, L2=0.001, registar 5, Brier+L2=0.970926849, max|grad|=6.106e-07
VALIDACIJA degree=2, L2=0.001, registar 6, Brier+L2=0.964955276, max|grad|=6.633e-07
VALIDACIJA degree=2, L2=0.001, registar 7, Brier+L2=0.941876358, max|grad|=3.039e-07
VALIDACIJA degree=2, L2=0.001
Srednji uslovni NLL: 17.274785289
Srednji broj pogodaka: 1.180769
Ciljevi sa nultom masom: 0
0 pogodaka: 61
1 pogodaka: 115
2 pogodaka: 63
3 pogodaka: 18
4 pogodaka: 3
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=2, L2=0.01, registar 1, Brier+L2=0.943431797, max|grad|=9.903e-07
VALIDACIJA degree=2, L2=0.01, registar 2, Brier+L2=0.981519962, max|grad|=8.178e-07
VALIDACIJA degree=2, L2=0.01, registar 3, Brier+L2=0.993065303, max|grad|=6.944e-07
VALIDACIJA degree=2, L2=0.01, registar 4, Brier+L2=0.993730155, max|grad|=9.145e-07
VALIDACIJA degree=2, L2=0.01, registar 5, Brier+L2=0.992791142, max|grad|=6.771e-07
VALIDACIJA degree=2, L2=0.01, registar 6, Brier+L2=0.995032596, max|grad|=7.952e-07
VALIDACIJA degree=2, L2=0.01, registar 7, Brier+L2=0.965154642, max|grad|=9.508e-07
VALIDACIJA degree=2, L2=0.01
Srednji uslovni NLL: 18.525379995
Srednji broj pogodaka: 1.200000
Ciljevi sa nultom masom: 0
0 pogodaka: 64
1 pogodaka: 104
2 pogodaka: 68
3 pogodaka: 24
4 pogodaka: 0
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.001, registar 1, Brier+L2=0.926303246, max|grad|=5.011e-07
VALIDACIJA degree=3, L2=0.001, registar 2, Brier+L2=0.959045939, max|grad|=8.830e-07
VALIDACIJA degree=3, L2=0.001, registar 3, Brier+L2=0.967830804, max|grad|=8.637e-07
VALIDACIJA degree=3, L2=0.001, registar 4, Brier+L2=0.971077323, max|grad|=9.286e-07
VALIDACIJA degree=3, L2=0.001, registar 5, Brier+L2=0.970117428, max|grad|=8.989e-07
VALIDACIJA degree=3, L2=0.001, registar 6, Brier+L2=0.965974727, max|grad|=4.326e-07
VALIDACIJA degree=3, L2=0.001, registar 7, Brier+L2=0.939340980, max|grad|=6.524e-07
VALIDACIJA degree=3, L2=0.001
Srednji uslovni NLL: 17.473770364
Srednji broj pogodaka: 1.157692
Ciljevi sa nultom masom: 0
0 pogodaka: 68
1 pogodaka: 103
2 pogodaka: 70
3 pogodaka: 18
4 pogodaka: 1
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
VALIDACIJA degree=3, L2=0.01, registar 1, Brier+L2=0.940085992, max|grad|=8.614e-07
VALIDACIJA degree=3, L2=0.01, registar 2, Brier+L2=0.975690729, max|grad|=9.611e-07
VALIDACIJA degree=3, L2=0.01, registar 3, Brier+L2=0.986702143, max|grad|=4.050e-07
VALIDACIJA degree=3, L2=0.01, registar 4, Brier+L2=0.988141696, max|grad|=5.529e-07
VALIDACIJA degree=3, L2=0.01, registar 5, Brier+L2=0.988065189, max|grad|=5.796e-07
VALIDACIJA degree=3, L2=0.01, registar 6, Brier+L2=0.984170109, max|grad|=9.180e-07
VALIDACIJA degree=3, L2=0.01, registar 7, Brier+L2=0.959089339, max|grad|=8.943e-07
VALIDACIJA degree=3, L2=0.01
Srednji uslovni NLL: 18.049119275
Srednji broj pogodaka: 1.200000
Ciljevi sa nultom masom: 0
0 pogodaka: 64
1 pogodaka: 104
2 pogodaka: 68
3 pogodaka: 24
4 pogodaka: 0
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0

Izabrano: degree=2, faza-parametara=21, L2=0.001
TEST TRENING, registar 1, Brier+L2=0.926962558, max|grad|=6.974e-07
TEST TRENING, registar 2, Brier+L2=0.959773937, max|grad|=9.658e-07
TEST TRENING, registar 3, Brier+L2=0.969680380, max|grad|=7.085e-07
TEST TRENING, registar 4, Brier+L2=0.971636528, max|grad|=9.459e-07
TEST TRENING, registar 5, Brier+L2=0.970437013, max|grad|=7.949e-07
TEST TRENING, registar 6, Brier+L2=0.964790617, max|grad|=6.101e-07
TEST TRENING, registar 7, Brier+L2=0.942170894, max|grad|=6.114e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.165354299
Srednji broj pogodaka: 1.298851
Ciljevi sa nultom masom: 0
0 pogodaka: 61
1 pogodaka: 104
2 pogodaka: 61
3 pogodaka: 29
4 pogodaka: 4
5 pogodaka: 2
6 pogodaka: 0
7 pogodaka: 0
CELA ISTORIJA, registar 1, Brier+L2=0.925932968, max|grad|=9.998e-07
CELA ISTORIJA, registar 2, Brier+L2=0.959451573, max|grad|=6.919e-07
CELA ISTORIJA, registar 3, Brier+L2=0.969731020, max|grad|=6.048e-07
CELA ISTORIJA, registar 4, Brier+L2=0.971544235, max|grad|=4.519e-07
CELA ISTORIJA, registar 5, Brier+L2=0.970729469, max|grad|=9.256e-07
CELA ISTORIJA, registar 6, Brier+L2=0.964299897, max|grad|=8.991e-07
CELA ISTORIJA, registar 7, Brier+L2=0.941983848, max|grad|=8.305e-07

Log legalne mase: -5.795705782
Log uslovne mase MAP-a: -13.759246180
Najveca promena raspodele kroz BV ulaze: 1.141843657e-01
Raspon NEXT kombinacije: 29
Udeo istorijskih raspona <= NEXT raspon: 39.827%

========================================================================
POREDJENJE TRI NEZAVISNA REZULTATA

========================================================================

loto7_4716_k89.csv NEXT: 6 10 19 25 29 31 34

loto7_4716_k89_loto_2980.csv NEXT: 12 14 17 18 27 33 34

loto7_4716_k89_loto_plus_1736.csv NEXT: 10 13 17 19 24 26 39
"""



"""
Q6-Q10	                            Osnovna ideja	                                                        Primena
Single Qubit Gates	                Rotacije, relativne faze i promene merne baze.	                        Učenje parametara jednokubitnih kapija. 
Multiple Qubits and Entanglement	Tenzorski proizvod, kontrolisane kapije i zapletenost.	                Učenje zapletenih stanja unutar šestokubitnih registara. 
Shor	                            Modularna eksponencijacija, procena faze i nalaženje reda.	            Posebna matematička veza sa istorijom kola. 
Bernstein-Vazirani	                Oporavak skrivenog niza iz linearnog binarnog orakla.	                Orakl izveden iz podataka koji zadovoljava taj uslov. 
Simon	                            Otkrivanje skrivene XOR simetrije funkcije.	                            Funkcija sa odgovarajućom strukturom preslikavanja. 


 6. Quantum_Single_Qubit_Gates.py
 7. Quantum_Entanglement.py
 8. Shor
 9. Bernstein-Vazirani
10. Simon

Za poslednja tri prvo treba definisati orakl iz istorije i proveriti uslove izvornog algoritma. 



--------
6. 

Quantum_Single_Qubit_Gates

Jednokubitna rotacija i fazna interferencija. 
Svaki šestokubitni registar ima proizvoljno stanje. 
Parametri kapija iz istorijskih prelaza kroz regresiju raspodele.



-------------
7. 

Quantum_Entanglement

Kontrolisane kapije i zapletenost. 
Između slojeva naučenih rotacija koriste se kontrolisane operacije. 
Provera redukovanih stanja pokazuje da li kolo zaista stvara zapletenost.

Rotacije zavisne od istorije pripremaju stanje, zatim CNOT lanac povezuje šest kubita, a završne rotacije menjaju mernu bazu. 
Provera zapletenosti preko čistoće redukovanih stanja i poređenje amplituda sa Qiskitom pre obuke.

Koristi naučene rotacije i CNOT lanac. 
Zapletenost se proverava unutar svakog šestokubitnog registra preko redukovanih matrica gustine. 
Redosled bitova prati Qiskit 1.4.

- smer i inverziju CNOT permutacija
- kompleksni gradijent kroz RY, RZ i CNOT
- faktor gradijenta Brierovog gubitka
- Bell i GHZ provere
- čistoća redukovanih stanja i Qiskit redosled bitova
- logaritamska normalizacija, globalni maksimum i hronološka podela 


Svaki kubit koristi niz RY → RZ → RY, koji pretvara relativnu fazu u promenu izlaznih verovatnoća. 
Prozor istorije bira se validacijom. 
Konačna kombinacija dobija se kao globalni maksimum raspodele nad svim legalnim izborima sedam brojeva.

Koristim sedam šestokubitnih registara i niz RY → RZ → RY na svakom kubitu. 
Prozore od jednog i tri prethodna kola biram validacijom. 
Optimizator uči parametre kapija iz cele trening istorije.


- izvodi verovatnoća po sva tri ugla RY → RZ → RY
- prefiksni i sufiksni proizvodi u gradijentu
- dimenzije parametara za oba prozora istorije
- redosled bitova prema Qiskitu
- logaritamska normalizacija i globalni izbor maksimuma
- granice treninga, validacije, testa i završnog učenja.



-------------
8. 

Shorov deo modularna eksponencijacija, QPE, inverzna QFT, procena faze i verizni ralomci za nalaženje reda. 

Orakl je \(f_a(x)=a^x\bmod 41\), uz izbor baze a hronološkom validacijom. 
Modul 41 omogućava modularni inverz za svaki broj od 1 do 39. 
QPE odredjuje red, a dodatni model uči amplitude koraka po modularnoj orbiti. 
Superpozicija periodičnih sopstvenih stanja modularnog operatora, čiju dimenziju određuje red dobijen QPE postupkom. 
Provera Shorove faktorizacije broja 15. 
Nalaženje reda koristiće 12 pomoćnih kubita i šest radnih kubita, a poređenje sa Qiskitom manju QPE konstrukciju.

Prediktivna dopuna uči superpoziciju periodičnih sopstvenih stanja modularnog operatora. 
Koristi modul 41, poslednje kolo kao početak orbite i celu istoriju prelaza za obuku. 
Bazu i broj koeficijenata bira validacijom.

QPE simulira 12 pomoćnih i šest radnih kubita. 
Manja QPE konstrukcija poredi se sa Qiskitom. 
Prediktivni deo koristi sedam šestokubitnih registara.



-------------
9.

Quantum_Bernstein_Vazirani 

Orakl, fazni povratni uticaj i Hadamard transformacije. 
Hadamard transformacije, linearni BV orakl i fazni povratni uticaj. 
Za Loto učim dodatnu fazu orakla iz istorijskih prelaza. 
Prethodno kolo je poznati ulaz. Buduća kombinacija se ne ugrađuje u orakl.

Koristim sedam registara sa po šest kubita i po jedan pomoćni kubit. 
Registri se simuliraju zasebno. 



----------------
10. 

Quantum_Simon

Sačuvaću orakl sa XOR periodom i rešavanje jednačina nad GF(2). 

Amplitude na parovima stanja \(x\) i \(x \oplus b\). 
Kandidati za XOR period biraju se iz treninga, a konačni model hronološkom validacijom. 
Simonovo kolo proverava period konstruisanog orakla. 

Dvostruki registar, orakl sa XOR periodom, Hadamard transformacije i rešavanje jednačina nad GF(2). 
Amplitude na Simonovim parovima stanja. 
Kandidati za period procenjuju se na treningu, a konačni model bira hronološkom validacijom.

Simonova provera koristi 12 kubita. 
Predikcija koristi sedam zasebnih registara sa po šest kubita. 
"""
