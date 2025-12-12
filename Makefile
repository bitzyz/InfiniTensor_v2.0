.PHONY : build clean check-infini format install-python test test-front

TYPE ?= Release
TEST ?= ON
# 平台参数（CUDA / ASCEND / CPU / ...）
PLATFORM ?= CPU
USE_CUDA ?= OFF
USE_ASCEND ?= OFF
USE_CAMBRICON ?= OFF
USE_METAX ?= OFF
USE_MOORE ?= OFF
USE_ILUVATAR ?= OFF
USE_SUGON ?= OFF
USE_KUNLUN ?= OFF
# 通信开关（ON / OFF）
COMM ?= OFF
FORMAT_ORIGIN ?=

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)

# InfiniCore 仓库地址
INFINICORE_URL = git@github.com:InfiniTensor/InfiniCore.git
INFINICORE_DIR = InfiniCore
CUR_DIR := $(shell pwd)


ifeq ($(PLATFORM), CPU)
    XMAKE_PLATFORM_FLAG = --cpu=y
else ifeq ($(PLATFORM), CUDA)
    XMAKE_PLATFORM_FLAG = --nv-gpu=y
	USE_CUDA = ON
else ifeq ($(PLATFORM), ASCEND)
    XMAKE_PLATFORM_FLAG = --ascend-npu=y
	USE_ASCEND = ON
else ifeq ($(PLATFORM), CAMBRICON)
    XMAKE_PLATFORM_FLAG = --cambricon-mlu=y
	USE_CAMBRICON = ON
else ifeq ($(PLATFORM), METAX)
    XMAKE_PLATFORM_FLAG = --metax-gpu=y
	USE_METAX = ON
else ifeq ($(PLATFORM), MOORE)
    XMAKE_PLATFORM_FLAG = --moore-gpu=y
	USE_MOORE = ON
else ifeq ($(PLATFORM), ILUVATAR)
    XMAKE_PLATFORM_FLAG = --iluvatar-gpu=y
	USE_ILUVATAR = ON
else ifeq ($(PLATFORM), SUGON)
    XMAKE_PLATFORM_FLAG = --sugon-dcu=y
	USE_SUGON = ON
else ifeq ($(PLATFORM), KUNLUN)
    XMAKE_PLATFORM_FLAG = --kunlun-xpu=y
	USE_KUNLUN = ON
else
    $(error Unknown PLATFORM=$(PLATFORM). Supported: CPU, CUDA, ASCEND, CAMBRICON, METAX, MOORE, ILUVATAR, SUGON, KUNLUN)
endif

CMAKE_OPT += -DUSE_CUDA=$(USE_CUDA)
CMAKE_OPT += -DUSE_ASCEND=$(USE_ASCEND)
CMAKE_OPT += -DUSE_CAMBRICON=$(USE_CAMBRICON)
CMAKE_OPT += -DUSE_METAX=$(USE_METAX)
CMAKE_OPT += -DUSE_MOORE=$(USE_MOORE)
CMAKE_OPT += -DUSE_ILUVATAR=$(USE_ILUVATAR)
CMAKE_OPT += -DUSE_SUGON=$(USE_SUGON)
CMAKE_OPT += -DUSE_KUNLUN=$(USE_KUNLUN)

# 通信参数
ifeq ($(COMM), ON)
    XMAKE_COMM_FLAG = --ccl=y
else
    XMAKE_COMM_FLAG = --ccl=n
endif

XMAKE_FLAGS = $(XMAKE_PLATFORM_FLAG) $(XMAKE_COMM_FLAG)

check-infini:
	@if [ -z "$$INFINI_ROOT" ]; then \
		echo "[INFO] INFINI_ROOT 未设置，开始拉取 InfiniCore ..."; \
		if [ ! -d "$(INFINICORE_DIR)" ]; then \
			git clone $(INFINICORE_URL); \
		fi; \
		echo "[INFO] 开始安装 InfiniCore (PLATFORM=$(PLATFORM), COMM=$(COMM)) ..."; \
		cd $(INFINICORE_DIR) && python scripts/install.py $(XMAKE_FLAGS); \
		echo "[INFO] 请运行 source ./start.sh 设置环境变量"; \
	else \
		echo "[INFO] 检测到 INFINI_ROOT=$$INFINI_ROOT"; \
	fi

# make build PLATFORM=CPU COMM=OFF
build: check-infini
	mkdir -p build/$(TYPE)
	cd build/$(TYPE) && cmake $(CMAKE_OPT) ../.. && make -j8

install-python: build
	cp build/$(TYPE)/pyinfinitensor*.so python/src/infinitensor
	pip install -e python/

clean:
	rm -rf build && rm -f python/src/infinitensor/*.so

test:
	cd build/$(TYPE) && make test

format:
	@python3 format.py $(FORMAT_ORIGIN)

test-front:
	pytest python/tests/
	