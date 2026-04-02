open Otograd

let usage () =
  Printf.eprintf
    "Usage:\n  dune exec ./bin/main.exe -- demo\n  dune exec ./bin/main.exe -- mnist <data-dir> [train-limit] [test-limit] [epochs] [learning-rate]\n";
  exit 1

let optional_argument arguments index =
  if List.length arguments > index then
    Some (List.nth arguments index)
  else
    None

let parse_int default = function
  | Some value -> int_of_string value
  | None -> default

let parse_float default = function
  | Some value -> float_of_string value
  | None -> default

let run_demo () =
  let open Value in
  let x = const ~label:"x" 0.5 in
  let y = const ~label:"y" (-0.25) in
  let z = add (sin x) (mul y (exp x)) in
  let loss = add (mul z z) (relu y) in
  let gradients = backward loss in
  Printf.printf "demo loss = %.6f\n" (data loss);
  Printf.printf "dloss/dx = %.6f\n" (grad gradients x);
  Printf.printf "dloss/dy = %.6f\n" (grad gradients y)

let run_mnist arguments =
  match arguments with
  | data_dir :: rest ->
      let train_limit = parse_int 512 (optional_argument rest 0) in
      let test_limit = parse_int 128 (optional_argument rest 1) in
      let epochs = parse_int 3 (optional_argument rest 2) in
      let learning_rate = parse_float 0.25 (optional_argument rest 3) in
      let split = Mnist.load_split ~train_limit ~test_limit data_dir in
      Printf.printf "loaded train=%d test=%d input_size=%d\n%!"
        (Array.length split.train) (Array.length split.test) split.input_size;
      let model = Nn.create_linear ~input_size:split.input_size () in
      let trained_model =
        Train.train_binary_linear ~epochs ~learning_rate model split.train
          split.test
      in
      let final_metrics = Train.evaluate_binary_linear trained_model split.test in
      Printf.printf "final test_loss=%.4f final test_acc=%.3f\n%!"
        final_metrics.loss final_metrics.accuracy
  | [] -> usage ()

let () =
  try
    match Array.to_list Sys.argv with
    | _ :: "demo" :: _ ->
        run_demo ()
    | _ :: "mnist" :: rest ->
        run_mnist rest
    | _ ->
        usage ()
  with
  | Failure message
  | Invalid_argument message
  | Sys_error message ->
      Printf.eprintf "error: %s\n" message;
      exit 1
