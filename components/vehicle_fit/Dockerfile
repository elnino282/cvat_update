ARG VEHICLE_FIT_BASE_IMAGE=cvat/server:dev
FROM ${VEHICLE_FIT_BASE_IMAGE}
WORKDIR /opt/vehicle-fit
COPY . /opt/vehicle-fit/vehicle_fit
ENV PYTHONUNBUFFERED=1 VEHICLE_FIT_BACKEND=geometry-only VEHICLE_FIT_PORT=8093
ENTRYPOINT ["python3", "-m", "vehicle_fit.service"]
