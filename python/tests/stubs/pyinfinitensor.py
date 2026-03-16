class DeviceType:
    CPU = "cpu"
    CUDA = "cuda"
    MLU = "mlu"
    ASCEND = "ascend"
    METAX = "metax"
    MOORE = "moore"
    ILUVATAR = "iluvatar"
    KUNLUN = "kunlun"
    HYGON = "hygon"


class Runtime:
    @staticmethod
    def setup(*args, **kwargs):
        return Runtime()


class Tensor:
    pass


class GraphBuilder:
    def __init__(self, runtime):
        self.runtime = runtime


class ShapeExpr:
    def __init__(self, shape):
        self.shape = shape


class StrideExpr:
    def __init__(self, stride):
        self.stride = stride


def dtype_from_string(dtype):
    return dtype
