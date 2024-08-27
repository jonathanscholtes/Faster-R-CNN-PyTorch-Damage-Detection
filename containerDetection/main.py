from coco_processor import COCOProcessor
from  container_data_processor import ContainerDataProcessor
from container_detection_trainer import ContainerDetectionTrainer
import os
import argparse

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

       

def train_model(annotations_path,source_image_path,results_path):
    """Function to train the model."""
    print("Training the model...")
    # Usage example
    trainer = ContainerDetectionTrainer(annotations_path=annotations_path, image_path=source_image_path,data_path=results_path,experiment_name="ContainerDetection",n_epochs=25)
    trainer.train_and_validate()
    print("Model training completed.")


def main(prep=False, train=False):
    """
    Arguments:
    prep (bool): If True, prepare the data.
    train (bool): If True, train the model.
    """

    annotations_path = 'annotations\\annotations.json'
    source_image_path = f"{os.path.dirname((os.path.dirname(os.path.abspath(__file__))))}\data\captured_images"
    coco_path = 'cocofiles\\container_coco.json'
    results_path = 'datafiles/data_retrain.json'

    try:

        if prep:
            prepare_data(source_image_path,coco_path,annotations_path,results_path)

        if train:
            train_model(annotations_path,source_image_path,results_path)


    except Exception as error:
        print('Exception: ' + error)  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Container Detection Training")
    
    # Set default values for the arguments
    parser.add_argument('--prep', action='store_true', default=False, help="Flag to prepare data (default: False)")
    parser.add_argument('--train', action='store_true', default=False, help="Flag to train the model (default: False)")
    
    args = parser.parse_args()

    # Use default values if arguments are not provided
    main(prep=args.prep, train=args.train)



