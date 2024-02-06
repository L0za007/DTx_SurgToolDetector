from PIL import Image
import numpy as np
import copy
from torchvision.datasets import CocoDetection


class CocoMi2cai(CocoDetection):

    def __init__(self, img_folder, annotation_file, remove_background=False, transformations=None, feature_extractor=None) -> None:
        super(CocoMi2cai, self).__init__(img_folder, annotation_file)
        self.transforms = transformations
        self.feature_extractor = feature_extractor
        if remove_background:
            print("The background will be removed")
            self.remove_background = TransWrapper(BG_remover())
        else:
            self.remove_background = None
            
    def __getitem__(self, idx:int):# -> tuple[Image.Image, dict]:
        if idx > self.__len__(): #check for a valid positive index
            raise IndexError(f"{idx} > {self.__len__()}")
        # get a sample (image and target) in COCO format Box format (X1, Y1, W, H)
        img, target = super(CocoMi2cai, self).__getitem__(idx)

        if self.remove_background is not None: 
            img, target = self.remove_background(img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.feature_extractor is not None: # Use feature extractor to set the samples into the format of the model (YOLO)
            # DETR format: resizing + normalization of both image and target. Box format (center X, center Y, W, H)
            # if the normalization if False the format of the box change to (X1, Y1, X2, Y2)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
            target = encoding["labels"][0] # remove batch dimension
            return pixel_values, target
        else :
            return img, target


class TransWrapper(object):
    """Wrapper to interface the COCOdataset format with the custom transformations for augmentation

    Parameters
    ----------
    transformations: object.Secuence
        Sequence of transformations to be applied on images and labels

    """
    def __init__(self, transformations) -> None:
        #Store the selected transformations
        self.tranformations = transformations

    def __call__(self, image, annot):
        """ Apply the pre-loaded transformations on the samples using self methods
        to go from the coco format to a handable format and back

        Parameters
        ----------
        image : Pil.Image
            Image (RGB format)
        annot : list(dict('bbox': list(x,y,w,h), **more_kargs))
            List of dictionaries for each label in a given image: Boundary boxe, object id, 
            boxe' area, image id.

        Returns
        -------
        Pil.Image
            Transformed image (RGB format)
        list(dict('bbox': list(x,y,w,h), **more_kargs))
            Updated annotations after transformations

        """
        ## 0 Do not change the format if there is not any transformation
        if self.tranformations == None:
            return image, annot
        ## 1 Get simplified version of the samples for the transformation methods
        rf_image, rf_annot = self.forward(image,annot)
        ## 2 Apply transformations over the extracted samples in step1
        rf_image, rf_annot = self.tranformations(rf_image, rf_annot)
        ## 3 Use the results from step 2 to update the annotation in the original format
        new_image, new_annot = self.backward(rf_image,rf_annot,copy.deepcopy(annot))
        
        return new_image, new_annot

    def forward(self, image:Image.Image, annot:dict): #-> tuple[np.ndarray, np.ndarray]:
        """Convert and image in Pil.Image format to numpy array and its list of annotations into
        a numpy array.

        Parameters
        ----------
        image : Pil.Image
            Image (RGB format)
        annot : list(dict('bbox': list(x,y,w,h), **more_kargs))
            List of dictionaries for each label in a given image: Boundary boxe, object id, 
            boxe' area, image id.

        Returns
        -------
        numpy.array 
            Image format RGB (H,W,C)
        numpy.array
            Annotation in a numpy array Nx5 array where N ist the number of labels in the image
            and axis 1 holds the location and tag (identifier) of the notations [x1,y1,x2,y2,tag]
        """
        # Setting image into numpy array format
        image = np.asarray(image)
        if len(annot) > 0: #if there are annotation for this image update their format
            boxes, tag = [], []
            for i,item in enumerate(annot): ## Extract the location of the boundary box and assing a tag
                boxes.append([item['bbox'][0],item['bbox'][1],
                item['bbox'][0]+item['bbox'][2],item['bbox'][1]+item['bbox'][3]])
                tag.append(i)
            ## Join boxes and tags into an Nx5 array where [x1,y1,x2,y2,tag]
            tag = np.array(tag)[:,None]
            boxes  = np.stack(boxes)
            array_boxes  = np.concatenate((boxes,tag), axis=1)
        else:
            array_boxes = np.array([[]])

        return image, array_boxes

    def backward(self, image:np.ndarray,boxes_and_tag:np.ndarray,annot_copy):#:list[dict]): -> tuple[Image.Image, list[dict]]:
        """Covert a numpy array (H,W,C) to pil.Image. Also, update the location of the annotations
        in the original list of labels (dictionaries)

        Parameters
        ----------
        image: numpy.array 
            Image format RGB (H,W,C)
        boxes_and_tag: numpy.array
            Annotation in a numpy array Nx5 array where N ist the number of labels in the image
            and axis 1 holds the location and tag (identifier) of the notations [x1,y1,x2,y2,tag]
        annot_copy: list(dict('bbox': list(x,y,w,h), **more_kargs))
            Copy of the original list of dictionaries for each label in a given image: Boundary boxe, object id, 
            boxe' area, image id. The order of the labels here should be the same as in the forward method
            to correctly match tags and key-i. 

        Returns
        -------
        Pil.Image
            Image (RGB format)
        list(dict('bbox': list(x,y,w,h), **more_kargs))
            List of dictionaries for each label in a given image: Boundary boxe, object id, 
            boxe' area, image id.
        """
        # Seting image back to the Pilow format
        image = Image.fromarray(image)
        # Empty list of labels in case the original list was empty or the annotations were loss during transformations
        new_annot = [] 
        if len(annot_copy)>0: #If originaly there were annotations try to updated annotations
            try: 
                # Set the annotations of the boxes in the original format
                boxes_and_tag[:,[2,3]] = boxes_and_tag[:,[2,3]]-boxes_and_tag[:,[0,1]]
                # Extract the tags as key and the annotation as value
                tags = {int(boxes_and_tag[i,-1]):boxes_and_tag[i,:-1] for i in range(boxes_and_tag.shape[0])}
            except:
                # If the labels were loss during transformation set tags to empty dict
                tags = {}
            for i,item in enumerate(annot_copy): # Update the location of the boundary box
                if i in tags.keys(): # Find the matches between the tags in the labels and the key i
                    item['bbox'] = list(tags[i])
                    new_annot.append(item)

        return image, new_annot
    
    def __str__(self) -> str:
        return str(self.tranformations)
    
    ############################## Transformation to remove black background
class BG_remover(object):
    """Remove the black background. 
    
    Returns
    -------
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box  
    """
    def __init__(self) -> None:
        pass

    def __call__(self, img, boxes):
        #print(img.shape)
        #print(boxes)
        #print("===========")
        # 1.- finde the borders of the real image
        half = int(img.shape[1]/2)
        mask = (img[:,:,0]>0) & (img[:,:,1]>0) & (img[:,:,2]>0)
        mask = mask.sum(axis=0)
        x_min, x_max = np.sum(mask[:half]<1), np.sum(mask[half:]<1)
        # 2.- Check that no more than 2/3 of the image is lost
        if x_min > half*.66: 
            x_min = 0
        if x_max > half*.66:
            x_max = img.shape[1]
        else:
            x_max = img.shape[1]-x_max   
        # 3.- Check integrity of the boxes
        if self._there_are_boxes(boxes):
            x1 = list(boxes[:,0].astype(int))
            x2 = list(boxes[:,2].astype(int))
            x1.append(x_min), x2.append(x_max)
            x_min, x_max = np.min(x1),np.max(x2)
            # 3.2.- adjust annotations
            boxes[:,0] = boxes[:,0] - x_min
            boxes[:,2] = boxes[:,2] - x_min
        # 4.- Crop the image
        img = img[:,x_min:x_max,:]
        #print(img.shape)
        #print(x_min, x_max)
        #print(boxes)
        return img, boxes
    
    def _there_are_boxes(self, boxes):
        """Verify boxes are present and in the correct format"""
        t1 = len(boxes.shape) == 2
        t2 = boxes.shape[1] == 5
        t3 = True
        try: 
            boxes[0][4]
        except:
            t3 = False
        return t1 and t2 and t3