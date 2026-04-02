open Otograd

let assert_close ?(epsilon = 1e-9) name expected actual =
  if Float.abs (expected -. actual) > epsilon then
    failwith
      (Printf.sprintf "%s mismatch: expected %.12f got %.12f" name expected
         actual)

let test_shared_node_gradient () =
  let open Value in
  let x = const 3. in
  let loss = mul x x in
  let gradients = backward loss in
  assert_close "shared.loss" 9. (data loss);
  assert_close "shared.dx" 6. (grad gradients x)

let test_matmul () =
  let open Value in
  let matrix =
    [| [| const 1.; const 2. |]; [| const 3.; const 4. |] |]
  in
  let vector = [| const 5.; const 6. |] in
  let output = Nn.matmul matrix vector in
  assert_close "matmul.0" 17. (data output.(0));
  assert_close "matmul.1" 39. (data output.(1))

let test_demo_graph_gradient () =
  let open Value in
  let x = const 0.5 in
  let y = const (-0.25) in
  let z = add (sin x) (mul y (exp x)) in
  let loss = add (mul z z) (relu y) in
  let gradients = backward loss in
  let expected_z = Stdlib.sin 0.5 +. (-0.25 *. Stdlib.exp 0.5) in
  let expected_loss = expected_z *. expected_z in
  let expected_dx =
    2. *. expected_z *. (Stdlib.cos 0.5 +. (-0.25 *. Stdlib.exp 0.5))
  in
  let expected_dy = 2. *. expected_z *. Stdlib.exp 0.5 in
  assert_close "demo.loss" expected_loss (data loss);
  assert_close "demo.dx" expected_dx (grad gradients x);
  assert_close "demo.dy" expected_dy (grad gradients y)

let () =
  test_shared_node_gradient ();
  test_matmul ();
  test_demo_graph_gradient ();
  Printf.printf "ok\n"
