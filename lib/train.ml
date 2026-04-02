type metrics = {
  loss : float;
  accuracy : float;
}

let progress_bar ~current ~total =
  let width = 28 in
  let safe_total = max total 1 in
  let filled = (current * width) / safe_total in
  let bar = Bytes.make width ' ' in
  for index = 0 to width - 1 do
    if index < filled then Bytes.set bar index '='
  done;
  Bytes.to_string bar

let show_training_progress ~epoch ~epochs ~current ~total ~loss ~accuracy =
  Printf.printf "\r[epoch %d/%d] [%s] %4d/%4d loss=%.4f acc=%.3f%!" epoch
    epochs (progress_bar ~current ~total) current total loss accuracy

let forward_binary_linear model pixels target =
  let input = Nn.const_vec ~prefix:"x" pixels in
  let probability = Value.sigmoid (Nn.linear model input) in
  let loss = Nn.binary_cross_entropy probability target in
  (probability, loss)

let train_sample model ~learning_rate (example : Mnist.example) =
  let probability, loss =
    forward_binary_linear model example.Mnist.pixels example.target
  in
  let gradients = Value.backward loss in
  let predicted_zero = Value.data probability >= 0.5 in
  let correct = predicted_zero = (example.label = 0) in
  let updated_model = Nn.update_linear model gradients ~learning_rate in
  (updated_model, Value.data loss, correct)

let evaluate_binary_linear model (dataset : Mnist.example array) =
  if Array.length dataset = 0 then invalid_arg "Dataset is empty";
  let total_loss = ref 0. in
  let total_correct = ref 0 in
  Array.iter
    (fun example ->
      let probability, loss =
        forward_binary_linear model example.Mnist.pixels example.target
      in
      total_loss := !total_loss +. Value.data loss;
      let predicted_zero = Value.data probability >= 0.5 in
      if predicted_zero = (example.label = 0) then incr total_correct)
    dataset;
  let size = float_of_int (Array.length dataset) in
  { loss = !total_loss /. size; accuracy = float_of_int !total_correct /. size }

let shuffle_in_place array random =
  for index = Array.length array - 1 downto 1 do
    let swap_index = Random.State.int random (index + 1) in
    let tmp = array.(index) in
    array.(index) <- array.(swap_index);
    array.(swap_index) <- tmp
  done

let train_binary_linear ?(epochs = 3) ?(learning_rate = 0.25) ?(seed = 42)
    model train_set test_set =
  if Array.length train_set = 0 then invalid_arg "Training split is empty";
  if Array.length test_set = 0 then invalid_arg "Test split is empty";
  let current = ref model in
  let random = Random.State.make [| seed |] in
  for epoch = 1 to epochs do
    let order = Array.init (Array.length train_set) (fun index -> index) in
    shuffle_in_place order random;
    let epoch_loss = ref 0. in
    let epoch_correct = ref 0 in
    Array.iteri
      (fun step index ->
        let next_model, loss, correct =
          train_sample !current ~learning_rate train_set.(index)
        in
        current := next_model;
        epoch_loss := !epoch_loss +. loss;
        if correct then incr epoch_correct;
        let current_step = step + 1 in
        let seen = float_of_int current_step in
        show_training_progress ~epoch ~epochs ~current:current_step
          ~total:(Array.length train_set) ~loss:(!epoch_loss /. seen)
          ~accuracy:(float_of_int !epoch_correct /. seen))
      order;
    Printf.printf "\n%!";
    let train_size = float_of_int (Array.length train_set) in
    let train_metrics =
      {
        loss = !epoch_loss /. train_size;
        accuracy = float_of_int !epoch_correct /. train_size;
      }
    in
    let test_metrics = evaluate_binary_linear !current test_set in
    Printf.printf
      "epoch %d train_loss=%.4f train_acc=%.3f test_loss=%.4f test_acc=%.3f\n%!"
      epoch train_metrics.loss train_metrics.accuracy test_metrics.loss
      test_metrics.accuracy
  done;
  !current
