LOCAL_SERVICE_PORT=5500
IMAGE_NAME=127.0.0.1:5000/vdinesh1990/bby-demo
MODEL_VERSION=v2

watch_docker:
	@watch -n 1 'docker ps -a'

build_model_image:
	s2i build --network host -E environment_rest -e "AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" -e "AWS_SECRET_ACCESS_KEY=${AWS_ACCESS_KEY_SECRET}" . seldonio/seldon-core-s2i-python3:0.13 $(IMAGE_NAME):$(MODEL_VERSION)

push_to_registry:
	docker push $(IMAGE_NAME):$(MODEL_VERSION)
