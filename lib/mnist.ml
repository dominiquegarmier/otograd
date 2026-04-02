type example = {
  pixels : float array;
  label : int;
  target : float;
}

type split = {
  train : example array;
  test : example array;
  input_size : int;
}

let read_u8 input = input_byte input

let read_be32 input =
  let b1 = read_u8 input in
  let b2 = read_u8 input in
  let b3 = read_u8 input in
  let b4 = read_u8 input in
  (((b1 lsl 24) lor (b2 lsl 16)) lor (b3 lsl 8)) lor b4

let load_labels ?limit path =
  let input = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in input)
    (fun () ->
      let magic = read_be32 input in
      if magic <> 2049 then
        invalid_arg
          (Printf.sprintf "Unexpected label magic number in %s: %d" path magic);
      let count = read_be32 input in
      let actual_count =
        match limit with
        | Some requested -> min requested count
        | None -> count
      in
      Array.init actual_count (fun _ -> read_u8 input))

let load_images ?limit path =
  let input = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in input)
    (fun () ->
      let magic = read_be32 input in
      if magic <> 2051 then
        invalid_arg
          (Printf.sprintf "Unexpected image magic number in %s: %d" path magic);
      let count = read_be32 input in
      let rows = read_be32 input in
      let cols = read_be32 input in
      let actual_count =
        match limit with
        | Some requested -> min requested count
        | None -> count
      in
      let images =
        Array.init actual_count (fun _ ->
            Array.init (rows * cols) (fun _ ->
                float_of_int (read_u8 input) /. 255.))
      in
      (rows, cols, images))

let downsample_2x2 pixels ~rows ~cols =
  if rows mod 2 <> 0 || cols mod 2 <> 0 then
    invalid_arg "downsample_2x2 expects even image dimensions";
  let new_rows = rows / 2 in
  let new_cols = cols / 2 in
  Array.init (new_rows * new_cols) (fun index ->
      let row = index / new_cols in
      let col = index mod new_cols in
      let base_row = row * 2 in
      let base_col = col * 2 in
      let i00 = (base_row * cols) + base_col in
      let i01 = i00 + 1 in
      let i10 = i00 + cols in
      let i11 = i10 + 1 in
      (pixels.(i00) +. pixels.(i01) +. pixels.(i10) +. pixels.(i11)) /. 4.)

let load ?limit ?(downsample = true) ~images_path ~labels_path () =
  let labels = load_labels ?limit labels_path in
  let rows, cols, images = load_images ?limit images_path in
  if Array.length labels <> Array.length images then
    invalid_arg "MNIST image and label counts do not match";
  Array.init (Array.length labels) (fun index ->
      let label = labels.(index) in
      let pixels =
        if downsample then
          downsample_2x2 images.(index) ~rows ~cols
        else
          images.(index)
      in
      let target = if label = 0 then 1. else 0. in
      { pixels; label; target })

let load_split ?(train_limit = 512) ?(test_limit = 128) directory =
  let train =
    load ~limit:train_limit
      ~images_path:(Filename.concat directory "train-images-idx3-ubyte")
      ~labels_path:(Filename.concat directory "train-labels-idx1-ubyte")
      ()
  in
  let test =
    load ~limit:test_limit
      ~images_path:(Filename.concat directory "t10k-images-idx3-ubyte")
      ~labels_path:(Filename.concat directory "t10k-labels-idx1-ubyte")
      ()
  in
  if Array.length train = 0 then invalid_arg "Training split is empty";
  { train; test; input_size = Array.length train.(0).pixels }
