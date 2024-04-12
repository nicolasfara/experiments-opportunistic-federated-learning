# Opportunistic Federated Learning - Experiments

> Provide a brief description of the experiments here.

## Reproduce the entire experiment

**WARNING**: re-running the whole experiment may take a very long time on a normal computer.

### Reproduce with containers (recommended)

1. Install docker and docker-compose
2. Run `docker-compose up`
3. The charts will be available in the `charts` folder.

### Reproduce natively

1. Install a Gradle-compatible version of Java.
  Use the [Gradle/Java compatibility matrix](https://docs.gradle.org/current/userguide/compatibility.html)
  to learn which is the compatible version range.
  The Version of Gradle used in this experiment can be found in the `gradle-wrapper.properties` file
  located in the `gradle/wrapper` folder.
2. Install the version of Python indicated in `.python-version` (or use `pyenv`).
3. Launch either:
    - `./gradlew runAllBatch` on Linux, MacOS, or Windows if a bash-compatible shell is available;
    - `gradlew.bat runAllBatch` on Windows cmd or Powershell;
4. Once the experiment is finished, the results will be available in the `data` folder. Run:
    - `pip install --upgrade pip`
    - `pip install -r requirements.txt`
    - `python process.py`
5. The charts will be available in the `charts` folder.

## Inspect a single experiment

Follow the instructions for reproducing the entire experiment natively, but instead of running `runAllBatch`,
run `runEXPERIMENTGraphics`, replacing `EXPERIMENT` with the name of the experiment you want to run
(namely, with the name of the YAML simulation file).

If in doubt, run `./gradlew tasks` to see the list of available tasks.

To make changes to existing experiments and explore/reuse,
we recommend to use the IntelliJ Idea IDE.
Opening the project in IntelliJ Idea will automatically import the project, download the dependencies,
and allow for a smooth development experience.

## Regenerate the charts

We keep a copy of the data in this repository,
so that the charts can be regenerated without having to run the experiment again.
To regenerate the charts, run `docker compose run --no-deps charts`.
Alternatively, follow the steps or the "reproduce natively" section,
starting after the part describing how to re-launch the simulations.

## Experiments description
The idea of the experiments is to verify the correct creation of opportunistic zones that follow the data distribution.
### Data
#### Classification
Here we plan to use some standard dataset for classification, such as:
- https://www.nist.gov/itl/products-and-services/emnist-dataset
- https://github.com/zalandoresearch/fashion-mnist
How? We can distribute the data following a Dirichlet distribution (TODO: verify if there are other distributions).
Idea: we can start with ~30 devices and then using more for extended minst
#### Regression
Here we can simulate several time series distribution (e.g., sinusoidal, linear, etc.) and distribute them among the devices.

### Metrics (for the experiments results)
- Loss/accuracy
- shape of the zones

TODO
- bring the leader election based on a metric
- distribute the data following some spatial distribution
- implement federated learning algorithm
