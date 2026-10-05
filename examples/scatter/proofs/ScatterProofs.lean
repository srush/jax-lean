import JaxLean.Stdlib
import examples.scatter.generated.ScatterMiddle
import examples.scatter.generated.ScatterColumn
import examples.scatter.generated.ScatterSelected
import examples.scatter.generated.ScatterAccumulate

namespace JaxLean.ScatterProofs
open ScatterJax

/-- A slice replacement, stated at the Python function boundary. -/
theorem replace_middle_spec (x : Tensor ℝ [4]) (values : Tensor ℝ [2]) :
    replace_middle x values = fun i =>
      if i.1 = 1 then values (0, ())
      else if i.1 = 2 then values (1, ()) else x i := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

theorem replace_column_spec (x : Tensor ℝ [2, 3]) (values : Tensor ℝ [2]) :
    replace_column x values = fun i =>
      if i.2.1 = 1 then values (i.1, ()) else x i := by
  funext ⟨i, j, u⟩
  cases u
  fin_cases i <;> fin_cases j <;> rfl

theorem replace_selected_spec (x : Tensor ℝ [4]) (values : Tensor ℝ [2]) :
    replace_selected x values = fun i =>
      if i.1 = 0 then values (0, ())
      else if i.1 = 2 then values (1, ()) else x i := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

/-- Repeated indices accumulate both contributions. -/
theorem accumulate_selected_spec (x : Tensor ℝ [4]) (values : Tensor ℝ [3]) :
    accumulate_selected x values = fun i =>
      if i.1 = 1 then x i + values (0, ()) + values (1, ())
      else if i.1 = 3 then x i + values (2, ()) else x i := by
  funext ⟨i, u⟩
  cases u
  fin_cases i <;> rfl

/-- The same function-boundary property, attached to the imported Jaxpr. -/
theorem accumulate_selected_certificate (x : Tensor ℝ [4]) (values : Tensor ℝ [3]) :
    Jaxpr.Program.eval (.cons x (.cons values .nil)) accumulate_selected_ir = fun i =>
      if i.1 = 1 then x i + values (0, ()) + values (1, ())
      else if i.1 = 3 then x i + values (2, ()) else x i := by
  rw [accumulate_selected_translation_correct, accumulate_selected_spec]

end JaxLean.ScatterProofs
