type vec = Value.t array
type mat = Value.t array array

type linear = {
  w : vec;
  b : Value.t;
}

let map2 name f left right =
  if Array.length left <> Array.length right then
    invalid_arg (name ^ " expects vectors of the same length");
  Array.mapi (fun index value -> f value right.(index)) left

let const_vec ?(prefix = "v") (values : float array) =
  Array.mapi
    (fun index value ->
      Value.const ~label:(Printf.sprintf "%s_%d" prefix index) value)
    values

let const_mat ?(prefix = "m") (rows : float array array) =
  Array.mapi
    (fun row_index row ->
      Array.mapi
        (fun col_index value ->
          Value.const
            ~label:(Printf.sprintf "%s_%d_%d" prefix row_index col_index)
            value)
        row)
    rows

let dot left right =
  Array.fold_left Value.add (Value.const 0.) (map2 "dot" Value.mul left right)

let matmul matrix vector =
  Array.map (fun row -> dot row vector) matrix

let binary_cross_entropy probability target =
  let open Value in
  let one = const 1. in
  let target_value = const target in
  let epsilon = const 1e-8 in
  let positive = mul target_value (log (add probability epsilon)) in
  let negative =
    mul (sub one target_value) (log (add (sub one probability) epsilon))
  in
  neg (add positive negative)

let create_linear ?(seed = 42) ~input_size () =
  if input_size <= 0 then invalid_arg "input_size must be positive";
  let random = Random.State.make [| seed |] in
  let scale = 1. /. Stdlib.sqrt (float_of_int input_size) in
  let weights =
    Array.init input_size (fun _ ->
        ((Random.State.float random 2.) -. 1.) *. scale)
  in
  { w = const_vec ~prefix:"w" weights; b = Value.const ~label:"b" 0. }

let input_size model =
  Array.length model.w

let validate_linear model input =
  if Array.length input <> input_size model then
    invalid_arg "input length does not match linear weight width"

let linear model input =
  validate_linear model input;
  Value.add (dot model.w input) model.b

let update_linear model gradients ~learning_rate =
  let w =
    Array.mapi
      (fun index weight ->
        let next =
          Value.data weight -. (learning_rate *. Value.grad gradients weight)
        in
        Value.const ~label:(Printf.sprintf "w_%d" index) next)
      model.w
  in
  let b =
    Value.const ~label:"b"
      (Value.data model.b -. (learning_rate *. Value.grad gradients model.b))
  in
  { w; b }
