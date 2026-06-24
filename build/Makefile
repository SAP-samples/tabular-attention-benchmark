.PHONY: build-fa2 build-fa3 build-sage build-all clean

WHEELS_DIR := $(CURDIR)/wheels
IMAGE_CU128 := tab-attn-build-cu128
IMAGE_CU130 := tab-attn-build-cu130

# Docker run flags: mount build script and output directory
DOCKER_RUN = docker run --rm \
	-v $(WHEELS_DIR):/output \
	-v $(CURDIR):/scripts:ro

# ── Docker images ─────────────────────────────────────────────────────────────

.docker-cu128: Dockerfile.cu128
	docker build -t $(IMAGE_CU128) -f Dockerfile.cu128 .
	@touch $@

.docker-cu130: Dockerfile.cu130
	docker build -t $(IMAGE_CU130) -f Dockerfile.cu130 .
	@touch $@

# ── Wheel targets ─────────────────────────────────────────────────────────────

build-fa2: .docker-cu128 | $(WHEELS_DIR)
	$(DOCKER_RUN) $(IMAGE_CU128) bash /scripts/build_fa2.sh

build-fa3: .docker-cu130 | $(WHEELS_DIR)
	$(DOCKER_RUN) $(IMAGE_CU130) bash /scripts/build_fa3.sh

build-sage: .docker-cu130 | $(WHEELS_DIR)
	$(DOCKER_RUN) $(IMAGE_CU130) bash /scripts/build_sage.sh

build-all: build-fa2 build-fa3 build-sage

$(WHEELS_DIR):
	mkdir -p $(WHEELS_DIR)

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(WHEELS_DIR) .docker-cu128 .docker-cu130
	-docker rmi $(IMAGE_CU128) $(IMAGE_CU130) 2>/dev/null
