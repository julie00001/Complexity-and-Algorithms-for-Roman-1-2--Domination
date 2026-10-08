SUPPLEMENTARY MATERIAL
Complexity and Algorithms for Roman [1,2]-Domination

This archive contains the code, synthetic graph instances, recorded results,
and reproduction instructions for the computational experiments and finite
verification in the manuscript. No external dataset is required.

START HERE

1. Read DATA_GUIDE.txt for the experimental design, data fields, and the exact
   correspondence between files and manuscript tables, statements, and figures.
2. Run the standard-library audit below to check file integrity, recorded
   certificates, CSV/JSON agreement, and all included table summaries.
3. Reconstruct the tables without optimization, or rerun the computations in a
   fresh directory. These are separate commands so that a reader can inspect
   the reported results without first installing an integer solver.

All commands below are run from the directory containing this README.

A. INSPECT AND RECONSTRUCT THE REPORTED TABLES

Python 3.12 is sufficient for these two commands; third-party packages and
solvers are not needed.

    python code/review.py audit
    python code/review.py tables

The first command checks MANIFEST.sha256, the 39 main instances, the eight
scaling instances, the finite-verification reports, and the seven included
CSV summaries/maps. The second writes freshly reconstructed CSV files to
results/tables/. It uses the published measurements; it does not rerun any
optimization. The files in tables/ contain these same reconstructed tables.

    tables/runtime.csv                 Runtime table (tab:runtime)
    tables/quality.csv                 Solution-ratio table (tab:quality)
    tables/scaling.csv                 Scalability table (tab:scaling)
    tables/experimental_design.csv     Family sizes and experimental purposes
    tables/instance_table_map.csv      Every benchmark instance to its table rows
    tables/appendix_coloring.csv       Appendix A.1 integer-program checks
    tables/appendix_cubic_reduction.csv Appendix A.3 cubic-reduction checks

B. INSTALL THE COMPUTATIONAL DEPENDENCIES

The recorded environment is Python 3.12.14, NetworkX 3.6.1, NumPy 2.3.5,
SciPy 1.18.1, and PuLP 3.3.0 on 64-bit Windows 10 with an Intel Core i7-12700F.
The main benchmark uses CBC 2.10.3 supplied with PuLP. The finite integer-
program checks use the HiGHS solver supplied with scipy.optimize.milp.
No separate highspy installation is required.

Create an environment using Python 3.12, then install the pinned requirements.

Windows PowerShell:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe code/reproduce.py all

Linux/macOS:

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python code/reproduce.py all

The code uses portable paths and subprocess calls. The full validation of
this archive was performed on the Windows environment stated above; execution
on other operating systems has not been validated here. To check the CBC
binary selected by PuLP, run the following with the environment's Python:

    python -c "import pulp; print(pulp.PULP_CBC_CMD().path)"

The specified PuLP distribution must provide an executable CBC binary.
Missing dependencies, a missing solver, or a solver process failure cause an
error; the scripts do not silently replace the solver or reuse published data.

C. RERUN THE COMPUTATIONS

Use the Python executable from the environment created above. For example,
replace 'python' below by '.venv\Scripts\python.exe' on Windows.

    python code/reproduce.py verification
    python code/reproduce.py benchmark
    python code/reproduce.py scaling

Alternatively, run all three groups sequentially:

    python code/reproduce.py all --output results-new

The output root contains the subdirectories verification/, benchmark/, and/or
scaling/. The runner refuses a nonempty group directory, ensuring a fresh run.
For a repeat run, choose a new --output directory. A failed or interrupted run
must also be retried in a fresh directory. Do not run Python with -O, because
the verification scripts use assertions to check mathematical conditions.

verification
  Runs the amplified split, amplified bipartite, and unamplified coloring
  checks, followed by the split-formula and approximation/recovery checks.
  Each finite integer program has a 45-second solver limit. The runner rejects
  an unverified solver result and compares the scientific results with the
  published records. It ignores elapsed times and solver message wording.

benchmark
  Generates all 39 small benchmark inputs, computes the structural solution,
  approximation, and baseline, and solves the direct integer formulation.
  CBC receives 10 seconds per integer model. The approximation is timed 20
  times per input. All generated inputs, objectives, bounds, and statuses are
  saved; intermediate checkpoints permit inspection of an interrupted run.
  The official runner nevertheless requires a fresh output directory on retry.

scaling
  Generates the eight larger inputs from the two polynomial subclasses and
  runs the structural and approximation algorithms three times per input.
  No direct integer model is run in this group.

Allow several minutes for a full run. Exact duration depends on the machine,
system load, and solver behavior. The runner checks generated inputs and the
reported exact/approximation objective values against data/. If the auxiliary
integer model cannot certify optimality within its time limit on a different
machine, the comparison fails visibly rather than claiming a full reproduction.

Timing values and the direct solver's time-limited incumbents, bounds, and
number of completed solves can vary. They are therefore retained in new
outputs but are not required to match the original timed records. In the
published main experiment, all 39 structural runs certify optimality; the
direct model independently certifies 19, and the other 20 stop at the limit.
Every newly returned target labeling is checked against the original
Roman [1,2] condition and its claimed weight by the computational scripts.

D. WHAT THE ARCHIVE CONTAINS

README.txt           Reproduction instructions and software/protocol information.
DATA_GUIDE.txt       Data description, design rationale, and manuscript mapping.
requirements.txt    Pinned third-party Python dependencies.
MANIFEST.sha256      SHA-256 checksums of every other distributed file.
code/                Five scientific scripts and three small execution/audit helpers.
data/                Published input/result records; treated as read-only references.
tables/              Seven CSV summaries/maps reconstructed from data/.

The main and scaling JSON files contain the complete input neighborhoods,
including their multiplicities, so the original graphs can be recovered
without rerunning random generation. Finite tests are specified completely
by the enumerations, named source graphs, and fixed seeds in their scripts.

The archive contains no manuscript drafts, research notes, pilot experiments,
review reports, local dependencies, solver binaries, or editing scripts.
Published numerical observations have been retained. The distributed code
uses separate output directories and a fresh-run wrapper; these packaging
changes do not change the mathematical algorithms or benchmark parameters.

The finite checks test implementations and concrete examples. The proofs in
the manuscript establish the results for entire graph classes. Timing data
describe the stated experimental setting, and are not complexity proofs.
