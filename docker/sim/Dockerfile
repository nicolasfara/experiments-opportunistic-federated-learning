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
RUN apt update && apt install -y python3 && apt install -y python3-pip python3-venv
# RUN CI=true ./gradlew runAllGraphic
#RUN rm -rf data
#RUN ./gradlew --stop
CMD ./gradlew runAllBatch
