from coco_processor import COCOProcessor
from  container_data_processor import ContainerDataProcessor
from container_detection_trainer import ContainerDetectionTrainer
from container_detection_inference import ContainerDetectionInference
import os
import argparse
import sys
from pathlib import Path



def prepare_data(source_image_path,coco_path,annotations_path,results_path):
    """Function to prepare the data."""
    print("Preparing data...")

   
    print("Step 1 - reshape into rotated bounding boxes")
    cocoProcessor = COCOProcessor(image_path=source_image_path, orig_coco_path=coco_path, annotation_path=annotations_path)
    cocoProcessor.process_annotations()
    cocoProcessor.shuffle_annotations()

    
    if cocoProcessor.validate_ids():
        print("Image and annotation IDs match!")
        cocoProcessor.save_to_file()

        print("Step 2 - Selective Search IOU Candidates")
        processor = ContainerDataProcessor(image_path=source_image_path,annotations_path=annotations_path,results_path=results_path)
        processor.process_dataset()

        print("Data preparation completed.")

    else:
        print("Mismatch between image and annotation IDs.")
        raise Exception("Mismatch between image and annotation IDs.")

       

def train_model(annotations_path,source_image_path,results_path,model_path):
    """Function to train the model."""
    print("Training the model...")
    # Usage example
    trainer = ContainerDetectionTrainer(annotations_path=annotations_path, image_path=source_image_path,data_path=results_path,experiment_name="ContainerDetection",n_epochs=25,model_path=model_path)
    trainer.train_and_validate()
    print("Model training completed.")




def inference_model(infer_source,infer_dest,model_path):
   
    inference = ContainerDetectionInference(model_path=model_path)
    image_files = []
    
    if os.path.isdir(infer_source):
        dir_path = Path(infer_source)
        for file in dir_path.rglob('*'):
            if file.is_file():
                image_files.append(file)
    else:
        image_files.append(infer_source)

    print(f"{len(image_files)} images for inference")

    results = inference.inference_model(image_files,infer_dest)

    print(results)
    return results



def main(prep=False, train=False, infer=False, infer_source="", infer_dest=""):
    """
    Arguments:
    prep (bool): If True, prepare the data.
    train (bool): If True, train the model.
    """

    annotations_path = 'annotations\\annotations.json'
    source_image_path = f"{os.path.dirname((os.path.dirname(os.path.abspath(__file__))))}\\data\\training\\captured_images"
    coco_path = 'cocofiles\\container_coco.json'
    results_path = 'datafiles/data.json'
    model_path="trained_models/frcnn_container.pt"

    try:

        if prep:
            prepare_data(source_image_path,coco_path,annotations_path,results_path)

        if train:
            train_model(annotations_path,source_image_path,results_path,model_path)

        if infer:
            inference_model(infer_source=infer_source,infer_dest=infer_dest, model_path=model_path)
        

    except Exception as error:
        print('Exception: ' + str(error))  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Container Detection Training")
    
    # Set default values for the arguments
    parser.add_argument('--prep', action='store_true', default=False, help="Flag to prepare data (default: False)")
    parser.add_argument('--train', action='store_true', default=False, help="Flag to train the model (default: False)")
    parser.add_argument('--infer', action='store_true', default=False, help="Flag to inference using file_path (default: False)")
    parser.add_argument('-infer_source', type=str, required='--infer' in sys.argv , help="file or folder path for source images")
    parser.add_argument('-infer_dest', type=str, required=False , help="folder path image results, will display json if blank")
    

    args = parser.parse_args()

    # Use default values if arguments are not provided
    main(prep=args.prep, train=args.train, infer=args.infer, infer_source=args.infer_source,infer_dest=args.infer_dest)



