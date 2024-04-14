from transformers import TrOCRProcessor, VisionEncoderDecoderModel








processor_handwritten = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model_handwritten = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')


# Save the models using save_pretrained method
# output_dir_printed = "./saved_model_printed"
output_dir_handwritten = "./saved_model_handwritten"

model_handwritten.save_pretrained(output_dir_handwritten)

processor_handwritten.save_pretrained(output_dir_handwritten)