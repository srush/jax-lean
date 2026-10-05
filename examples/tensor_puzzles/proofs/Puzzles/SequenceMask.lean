import JaxLean.Stdlib
import examples.tensor_puzzles.generated.PuzzlesSequenceMask

set_option maxRecDepth 4096
set_option maxHeartbeats 2000000
namespace JaxLean.Puzzles.SequenceMask

theorem matches_loop («values» : Tensor ℝ [2, 3]) («length» : Tensor Int32 [2]) :
    Array.sequence_mask «values» «length» = Loop.sequence_mask «values» «length» := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;>
    simp only [Array.fn__where, Array.sequence_mask, Loop.fn__where, Loop.loop_sequence_row,
      Loop.sequence_mask, Tensor.map, Tensor.scalar, Tensor.reindex, Tensor.scatterSet_vector,
      Tensor.scatterSet_matrix]
  all_goals norm_num [Fin.ext_iff, Bool.cond_decide, decide_eq_true_iff]

theorem certificate («values» : Tensor ℝ [2, 3]) («length» : Tensor Int32 [2]) :
    Jaxpr.Program.eval (.cons «values» (.cons «length» .nil)) Array.sequence_mask_ir =
      Jaxpr.Program.eval (.cons «values» (.cons «length» .nil)) Loop.sequence_mask_ir := by
  rw [Array.sequence_mask_translation_correct, Loop.sequence_mask_translation_correct]
  exact matches_loop «values» «length»

end JaxLean.Puzzles.SequenceMask
