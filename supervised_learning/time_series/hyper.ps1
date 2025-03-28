<#
: --data_dir: Directory with preprocessed data (default: "preprocessed_data")
: --model_type: Type of RNN model to use (choices: "lstm" (default), "gru", "bidirectional")
: --epochs: Number of training epochs (default: 50)
: --batch_size: Batch size for training (default: 32)
: --output: Output directory for model and results (default: "model_output")
#>

$models = @("lstm", "gru", "bidirectional")
$epochs = @(50, 100)
$batchSizes = @(32, 64, 128)

foreach ($model in $models) {
    foreach ($epoch in $epochs) {
        foreach ($batch in $batchSizes) {
            $output = $model.ToUpper() + "_" + $epoch + "_" + $batch
            python forecast_btc.py --model_type $model --epochs $epoch --batch_size $batch --output $output
        }
    }
}
