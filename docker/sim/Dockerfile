FROM eclipse-temurin:17
RUN mkdir /experiment
WORKDIR /experiment
COPY effects effects
COPY gradle gradle
COPY src src
COPY docker docker
COPY python python
COPY *.kts ./
COPY *.properties ./
COPY requirements.txt ./
RUN sed -i '/alias(libs.plugins.gitSemVer)/d' build.gradle.kts
COPY gradlew* ./
RUN apt update && apt install -y python3 && apt install -y python3-pip
RUN pip install requirements.txt
RUN cd python && python3 setup.py sdist bdist_wheel
RUN pip install -e python
RUN CI=true ./gradlew runAllGraphic
RUN rm -rf data
RUN ./gradlew --stop
CMD ./gradlew runAllBatch
