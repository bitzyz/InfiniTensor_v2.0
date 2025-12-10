.PHONY : build clean check-infini

TYPE ?= Release
TEST ?= ON
# 平台参数（CUDA / ASCEND / CPU / ...）
PLATFORM ?= CPU
# 通信开关（ON / OFF）
COMM ?= OFF

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
else ifeq ($(PLATFORM), ASCEND)
    XMAKE_PLATFORM_FLAG = --ascend-npu=y
else ifeq ($(PLATFORM), CAMBRICON)
    XMAKE_PLATFORM_FLAG = --cambricon-mlu=y
else ifeq ($(PLATFORM), METAX)
    XMAKE_PLATFORM_FLAG = --metax-gpu=y
else ifeq ($(PLATFORM), MOORE)
    XMAKE_PLATFORM_FLAG = --moore-gpu=y
else ifeq ($(PLATFORM), ILUVATAR)
    XMAKE_PLATFORM_FLAG = --iluvatar-gpu=y
else ifeq ($(PLATFORM), SUGON)
    XMAKE_PLATFORM_FLAG = --sugon-dcu=y
else ifeq ($(PLATFORM), KUNLUN)
    XMAKE_PLATFORM_FLAG = --kunlun-xpu=y
else
    $(error Unknown PLATFORM=$(PLATFORM). Supported: CPU, CUDA, ASCEND, CAMBRICON, METAX, MOORE, ILUVATAR, SUGON, KUNLUN)
endif

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

clean:
	rm -rf build

test:
	cd build/$(TYPE) && make test
