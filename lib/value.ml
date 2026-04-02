type unary_op =
  | Sin
  | Exp
  | Log
  | Neg
  | Tanh
  | Relu

type binary_op =
  | Add
  | Sub
  | Mul
  | Div

type op =
  | Constant
  | Unary of unary_op
  | Binary of binary_op

type t = {
  id : int;
  data : float;
  op : op;
  parents : t list;
  label : string option;
}

type gradients = (int, float) Hashtbl.t

let fresh_id =
  let counter = ref 0 in
  fun () ->
    let id = !counter in
    incr counter;
    id

let make ?label data op parents = { id = fresh_id (); data; op; parents; label }

let const ?label data = make ?label data Constant []
let data value = value.data
let id value = value.id
let label value = value.label
let parents value = value.parents

let unary op f x = make (f x.data) (Unary op) [ x ]
let binary op f a b = make (f a.data b.data) (Binary op) [ a; b ]

let add a b = binary Add ( +. ) a b
let sub a b = binary Sub ( -. ) a b
let mul a b = binary Mul ( *. ) a b
let div a b = binary Div ( /. ) a b
let neg x = unary Neg (fun value -> -. value) x
let sin x = unary Sin Stdlib.sin x
let exp x = unary Exp Stdlib.exp x
let log x = unary Log Stdlib.log x
let tanh x = unary Tanh Stdlib.tanh x
let relu x = unary Relu (fun value -> if value > 0. then value else 0.) x

let sigmoid x =
  div (const 1.) (add (const 1.) (exp (neg x)))

let topo_sort root =
  let seen = Hashtbl.create 256 in
  let order = ref [] in
  let rec visit value =
    if not (Hashtbl.mem seen value.id) then (
      Hashtbl.add seen value.id ();
      List.iter visit value.parents;
      order := value :: !order)
  in
  visit root;
  List.rev !order

let accumulate gradients value delta =
  let current =
    match Hashtbl.find_opt gradients value.id with
    | Some existing -> existing
    | None -> 0.
  in
  Hashtbl.replace gradients value.id (current +. delta)

let grad gradients value =
  match Hashtbl.find_opt gradients value.id with
  | Some gradient -> gradient
  | None -> 0.

let backward loss =
  let gradients = Hashtbl.create 256 in
  accumulate gradients loss 1.;
  let order = topo_sort loss in
  List.iter
    (fun value ->
      let upstream = grad gradients value in
      match (value.op, value.parents) with
      | Constant, _ -> ()
      | Unary Sin, [ x ] ->
          accumulate gradients x (upstream *. Stdlib.cos x.data)
      | Unary Exp, [ x ] ->
          accumulate gradients x (upstream *. value.data)
      | Unary Log, [ x ] ->
          accumulate gradients x (upstream /. x.data)
      | Unary Neg, [ x ] ->
          accumulate gradients x (-. upstream)
      | Unary Tanh, [ x ] ->
          accumulate gradients x (upstream *. (1. -. (value.data *. value.data)))
      | Unary Relu, [ x ] ->
          let local = if x.data > 0. then 1. else 0. in
          accumulate gradients x (upstream *. local)
      | Binary Add, [ left; right ] ->
          accumulate gradients left upstream;
          accumulate gradients right upstream
      | Binary Sub, [ left; right ] ->
          accumulate gradients left upstream;
          accumulate gradients right (-. upstream)
      | Binary Mul, [ left; right ] ->
          accumulate gradients left (upstream *. right.data);
          accumulate gradients right (upstream *. left.data)
      | Binary Div, [ left; right ] ->
          accumulate gradients left (upstream /. right.data);
          accumulate gradients right
            (upstream *. (-. left.data /. (right.data *. right.data)))
      | _ ->
          invalid_arg "Value.backward: malformed node")
    (List.rev order);
  gradients
