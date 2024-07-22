# flake8: noqa: B950
from typing import List, Optional, Tuple

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import (
    LearnedHeuristicDecision,
)


class MixedMMA100(LearnedHeuristicDecision):
    def __init__(self) -> None:
        self.choices: List[Choice] = []
        self.fill_choices()

    def check_precondition(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> bool:
        return (
            metadata.name == self.get_name()
            and metadata.shared_memory == 166912
            and str(metadata.device_capa) == "(8, 0)"
        )

    def get_confidence_threshold(self) -> float:
        return 0.0

    def get_choice(self, idx: int) -> Optional[str]:
        if idx < len(self.choices):
            return self.choices[idx]
        return None

    def fill_choices(self) -> None:
        self.choices.append("extern_fallback_mixed_mm")
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=128_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=2"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=2"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=256_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=256_BLOCK-N=128_numstages=5_numwarps=8"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8"
        )
        self.choices.append(
            "type=triton_BLOCK-M=16_BLOCK-K=64_BLOCK-N=64_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=2_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=32_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=128_numstages=4_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=32_numstages=5_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=128_BLOCK-N=64_numstages=5_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=128_numstages=4_numwarps=8"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=32_BLOCK-N=64_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=3_numwarps=4"
        )
        self.choices.append(
            "type=triton_BLOCK-M=64_BLOCK-K=64_BLOCK-N=128_numstages=5_numwarps=8"
        )

    def get_name(self) -> str:
        return "mixed_mm"

    def get_best_choices(self, context: AHContext) -> Optional[List[Tuple[float, int]]]:
        if str(context.get_value("1LEQmLEQ16")) != "True":
            if context.get_value("m") <= 32.5:
                if context.get_value("n") <= 6976.0:
                    if context.get_value("n") <= 3520.0:
                        if context.get_value("m*n") <= 37632.0:
                            return None
                        else:
                            return [(1.000, 13)]
                    else:
                        if context.get_value("m*k") <= 452352.0:
                            return [(0.590, 13), (0.256, 8), (0.103, 7), (0.051, 11)]
                        else:
                            return [(0.778, 8), (0.222, 13)]
                else:
                    if context.get_value("k*n") <= 102776832.0:
                        if context.get_value("n") <= 14656.0:
                            return [(1.000, 11)]
                        else:
                            return [(0.889, 11), (0.111, 13)]
                    else:
                        return [(1.000, 11)]
            else:
                if context.get_value("m*n") <= 446464.0:
                    if context.get_value("m*n") <= 223424.0:
                        if context.get_value("mat1_stride_0") <= 3968.0:
                            return None
                        else:
                            return None
                    else:
                        if context.get_value("m*n") <= 346112.0:
                            return [(0.960, 16), (0.040, 7)]
                        else:
                            return [(0.750, 16), (0.136, 14), (0.114, 7)]
                else:
                    if str(context.get_value("33LEQmLEQ64")) != "True":
                        if context.get_value("n") <= 6976.0:
                            return [(1.000, 14)]
                        else:
                            return [
                                (0.753, 2),
                                (0.222, 1),
                                (0.015, 7),
                                (0.007, 16),
                                (0.004, 12),
                            ]
                    else:
                        if context.get_value("n") <= 13888.0:
                            return [(0.710, 14), (0.275, 21), (0.014, 12)]
                        else:
                            return [
                                (0.374, 19),
                                (0.339, 20),
                                (0.106, 21),
                                (0.101, 16),
                                (0.066, 17),
                                (0.009, 14),
                                (0.004, 18),
                            ]
        else:
            if context.get_value("n") <= 3520.0:
                if context.get_value("arith_intensity") <= 3.994754433631897:
                    if str(context.get_value("mat2_dtype")) != "torch.uint8":
                        if context.get_value("m*k") <= 18944.0:
                            return [(0.577, 5), (0.423, 6)]
                        else:
                            return [(0.988, 5), (0.012, 6)]
                    else:
                        if context.get_value("arith_intensity") <= 2.9899919033050537:
                            return None
                        else:
                            return None
                else:
                    if context.get_value("arith_intensity") <= 7.956453561782837:
                        if context.get_value("k*n") <= 9244032.0:
                            return [(0.822, 5), (0.178, 6)]
                        else:
                            return [(0.977, 5), (0.023, 0)]
                    else:
                        if context.get_value("m*k") <= 978944.0:
                            return [(1.000, 5)]
                        else:
                            return [(0.971, 5), (0.029, 0)]
            else:
                if context.get_value("n") <= 13632.0:
                    if context.get_value("n") <= 6976.0:
                        return [(1.000, 6)]
                    else:
                        if context.get_value("k") <= 3968.0:
                            return [
                                (0.617, 3),
                                (0.111, 5),
                                (0.099, 7),
                                (0.086, 9),
                                (0.062, 6),
                                (0.025, 8),
                            ]
                        else:
                            return [
                                (0.779, 8),
                                (0.119, 5),
                                (0.053, 7),
                                (0.035, 6),
                                (0.013, 3),
                            ]
                else:
                    if context.get_value("k*n") <= 39518208.0:
                        return [
                            (0.385, 4),
                            (0.327, 3),
                            (0.192, 6),
                            (0.038, 7),
                            (0.038, 10),
                            (0.019, 5),
                        ]
                    else:
                        if context.get_value("n") <= 20800.0:
                            return [
                                (0.821, 6),
                                (0.121, 7),
                                (0.029, 4),
                                (0.014, 5),
                                (0.007, 3),
                                (0.007, 8),
                            ]
                        else:
                            return [
                                (0.530, 7),
                                (0.386, 6),
                                (0.046, 8),
                                (0.021, 3),
                                (0.015, 4),
                                (0.002, 5),
                            ]
