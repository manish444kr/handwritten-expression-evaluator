import numpy as np
import cv2
import matplotlib.pyplot as plt

class contour_detection() :
    
    def __init__(self,image):
        self.im= image
        
        # brighten image
        # self.im is illuminated image
        self.im= self.illumination(self.im)

        # preprocess
        # self.new_img : preprocessed image
        self.new_img = self.preprocessing(self.im)

        # cropped image array and outputs tight crop
        # self.new_img : tightly_cropped image
        # self.coor : ??

        self.new_img, self.coor= self.tight_crop(self.new_img) 

        # sending original image according to tight crop coordiantes
        # self.cont : list of contours
        # self.cont_coor : list of co-ordinates of corresponding contours
        self.conts, self.contour_coor= self.contours(self.new_img,save=False)

        # 
        # self.ind_images : list of individual images out of tightly cropped images
        self.ind_images = self.extractIndividualImages(self.new_img,self.contour_coor)

        # Finding contours in eav=ch individual image
        self.all_contours_img = []
        self.all_contours_coor = []
        self.contours_len = []
        indx = 0
        name_indx = 1	

        for i in range(len(self.ind_images)) :
            ct , ct_coor = self.contours(self.ind_images[i],save=True,indx = indx)
            name_indx += len(ct)
            self.contours_len.append(name_indx)
            ct_adjusted , ct_coor_adjusted = self.adjustContour(ct_coor,self.ind_images[i],name_indx)
            #print('ind_img' , len(self.ind_images[i]) , self.ind_images[i][0].shape)
            ct_padded = self.pad_image(ct_adjusted)
            ct_sorted , ct_coor_sorted = self.sortContours(ct_padded,ct_coor_adjusted,indx) 
            ct_sorted , ct_coor_sorted = self.removeRedundant(ct_sorted , ct_coor_sorted)
            self.all_contours_img.append(ct_sorted)
            self.all_contours_coor.append(ct_coor_sorted)
            

        #encapsulating contours to solve divison operator problem
        #self.images, self.new_contour_coor= self.adjustContour(self.contour_coor,self.new_img)

        # pad images
        #self.new_images=self.pad_image(self.images)

        # sorting the contours according to coordinates
        #self.new_conts,self.new_contour_coor=self.sort_contour(self.new_images,self.new_contour_coor)
        
    # function to change brightness of image
    def illumination(self,orig):

        #image in hsv format
        im=cv2.cvtColor(orig,cv2.COLOR_BGR2HSV)

        # brightness value
        value=im[...,2]
        im[...,2]=np.where(value<140,185,185)

        im=cv2.cvtColor(im,cv2.COLOR_HSV2BGR)
        
        # save image
        cv2.imwrite(r"C:/Users/rajneesh/Downloads/new project/detected/illumination.jpg",im)

        return im
    
    # to preprocess image for tight crop
    def preprocessing(self,orig) :

        # grayscale
        orig_gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)

        # blur
        img_blur=cv2.GaussianBlur(orig_gray,(7,7),0)

        #p ixels with value below 150 are turned black (0) and those with higher value are turned white (255)
        _,img_thresh=cv2.threshold(img_blur,150,255,cv2.THRESH_BINARY)

        # save image
        cv2.imwrite(r"C:/Users/rajneesh/Downloads/new project/detected/preprocess.jpg",img_thresh)    

        return img_thresh
    
    # sending threshold image and getting cropped contour enclosing equation
    def tight_crop(self,img) :

        #image area
        area=img.shape[0]*img.shape[1]

        # contour coordinates
        # x,x+w,y,y+h
        min_x=img.shape[0]
        min_y=img.shape[1]
        max_x=0
        max_y=0

        # finding contours
        _,cnts,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #finding the contour enclosing the equation
        for cnt in cnts:
            x,y,w,h =  cv2.boundingRect(cnt)
            if w*h>area*0.002 and w*h != area:

                min_x=min(x,min_x)
                min_y=min(y,min_y)
                max_x=max(x+w,max_x)
                max_y=max(y+h,max_y)

        # save image in list
        new_image=img[min_y-15:max_y+15,min_x-15:max_x+15]
        
        # save contour coordinates
        coor=[min_x,min_y,max_x,max_y]

        # save image to local
        cv2.imwrite(r"C:/Users/rajneesh/Downloads/new project/detected/tight_crop.jpg",new_image)    

        return new_image,coor

    
    # contours
    def contours(self,orig_img,save,indx=0):

        # img=orig_img[coor[1]-10:coor[3]+10,coor[0]-10:coor[2]+10]
        img=orig_img

        #thresh
        _,img_thresh=cv2.threshold(orig_img,120,250,cv2.THRESH_BINARY)  

        #canny
        img_canny=cv2.Canny(img_thresh,240,250)

        # find contours
        _,cnts,_= cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # area of image
        area=img_canny.shape[0]*img_canny.shape[1]

        # save detected contours in a list
        cont=[]

        #to save coordinates of contours
        contour_coor={}

        
        for cnt in cnts:

            x,y,w,h=cv2.boundingRect(cnt)

            if w*h>0.002*area and w*h!=area:

                #contour coordinates
                contour_coor[indx]=[x,x+w,y,y+h]

                #append contours in array
                cont.append(img[y:y+h,x:x+w])

                #save contour images
                crop_rect=img[y:y+h,x:x+w]
                
                indx+=1
                
                #save contour to local
                if save :
                    cv2.imwrite(r"C:/Users/rajneesh/Downloads/new project/detected/"+str(indx)+".jpg",crop_rect)

        # save images to local
        if save :
        	cv2.imwrite(r'C:/Users/rajneesh/Downloads/new project/detected/threshold.jpg',img_thresh)
        	cv2.imwrite(r'C:/Users/rajneesh/Downloads/new project/detected/canny.jpg',img_canny)
        
        return cont,contour_coor
    
    # divide solution
    def adjustContour(self,contour_coor , image,head_start) :

        new_cont_coor = {}

        # list to save images
        new_images=[]
        dict_images={}

        # to save the overlapping contours
        marked_index=[]

        # contour_coor is a dictionary that contains the values in the format of [x , x+w , y , y+h]
        for indx,item in contour_coor.items() :

            x = item[1] - item[0] # x+w - x
            y = item[3] - item[2] # y+h - y

            
            
            ## Remove the if condition and do the if part for all of the contours

            
            if x > 2*y :
                diff=image.shape[1]-y
                side = diff//2
                new_cont_coor[indx] = [item[0] , item[1] , item[2]-side , item[3]+side]
                marked_index.append(indx)

            else :
                new_cont_coor[indx] = item
            
            '''
            diff=image.shape[1]-y
            side = diff//2
            new_cont_coor[indx] = [item[0] , item[1] , item[2]-side , item[3]+side]
            marked_index.append(indx)
            '''

            #contour coordinates
            a=new_cont_coor[indx][0]
            b=new_cont_coor[indx][1]
            c=new_cont_coor[indx][2]
            d=new_cont_coor[indx][3]


            #save contour images
            crop_rect=image[c:d,a:b]

            # creating a dictionary
            dict_images[indx]=crop_rect
        
        #print(marked_index)
        #print('dict_images.keys' , list(dict_images.keys()))

        


        # removing redundant contours
        '''
        for i in range(len(dict_images)):#marked_index)):

            del_keys=[]
            #print(marked_index)

            left_limit=new_cont_coor[marked_index[i+head_start-1]][0]
            right_limit=new_cont_coor[marked_index[i]+head_start-1][1]

            #detect redundant contous
            for key in new_cont_coor.keys() :

                # centre coordinates of the present contour
                x_mid=(new_cont_coor[key][0]+new_cont_coor[key][1])/2

                # same contours
                if(key==marked_index[i]):
                    continue
                
                # to remove overlapping contours
                elif(left_limit<=x_mid and x_mid<=right_limit):
                    del_keys.append(key)

            #delete redundant contours
            for key in del_keys:
                del dict_images[key]
                del new_cont_coor[key]

            temp_coor={}
            temp_images={}
            count=0
            for key in new_cont_coor.keys() :
                temp_coor[count]=new_cont_coor[key]
                temp_images[count]=dict_images[key]
                count+=1

            new_cont_coor=temp_coor
            dict_images=temp_images
		'''
        new_images=[]
        for k,v in dict_images.items():
            #print(k , end = ' , ')
            new_images.append(v)
        



        return new_images,new_cont_coor           

    # pad crop image
    def pad_image(self,images):

        new_images=[]
        for im in images:
            new_im=cv2.copyMakeBorder(im,top=40,bottom=40,left=40,right=40,borderType=cv2.BORDER_CONSTANT,value=[255,255,255])
            new_images.append(new_im)
        return new_images
    
    #sorting contours
    def extractIndividualImages(self,cropped_img,cropped_cont_coor):

        orig_img = cropped_img
        cont = cropped_cont_coor

        cont_coor = list(cropped_cont_coor.values())
        cont_name = list(cropped_cont_coor.keys())

        #sorting according to y co-ordinates
        for i in range(len(cont)-1):
            for j in range(len(cont)-i-1):
                if(cont_coor[j][2] > cont_coor[j+1][2]):
                    cont_coor[j],cont_coor[j+1] = cont_coor[j+1],cont_coor[j]
                    orig_img[j],orig_img[j+1] = orig_img[j+1],orig_img[j]

        indx = 1

        exps = []#containing the contours in row-wise order
        new_img = []#containing the images in row-wise order
        prev = 0#counter
        ind_images = []

        #sorting contours in row-wise order

        for i in range(len(cont)-1):
            if cont_coor[i][3] < cont_coor[i+1][2]:
                row_cont = cont_coor[prev:i+1]#all the same row contours in single list
                row_img = orig_img[prev:i+1]
                exps.append(row_cont)
                new_img.append(row_img)
                prev = i+1

        #appending left out contours
        new_img.append(orig_img[prev:len(orig_img)])
        exps.append(cont_coor[prev:len(cont)])
        #print('exps' , exps)
        '''
        #sorting according to x-coordinates
        for i in range(len(new_img)):#first row of images
            for k in range(len(new_img[i])-1):
                for j in range(len(new_img[i])-k-1):
                    if exps[i][j][0] > exps[i][j+1][0] :
                        new_img[i][j] , new_img[i][j+1] = new_img[i][j+1] , new_img[i][j]
                        exps[i][j] , exps[i][j+1] = exps[i][j+1] , exps[i][j]
		'''
		#N = len(exps)
        for img in exps:
        	max_x = max_y = -1
        	min_x = min_y = 10**6
        	for cnt in img :
        		if min_x > cnt[0] :
        			min_x = cnt[0]
        		if max_x < cnt[1] :
        			max_x = cnt[1]
        		if min_y > cnt[2] :
        			min_y = cnt[2]
        		if max_y < cnt[3] :
        			max_y = cnt[3]	
        		'''
	            min_x = min(list(x[img][cnt][0] for x in exps))
	            max_x = max(list(x[img][cnt][1] for x in exps))
	            min_y = min(list(y[img][cnt][2] for y in exps))
	            max_y = max(list(y[img][cnt][3] for y in exps))
	            '''
	        ind_images.append(cropped_img[min_y:max_y , min_x:max_x])       
        return ind_images

    def sortContours(self,img,coor,head_start) :
        head_start = 0
        n = len(img)
        #print('coor' ,coor)
        for i in range(n): 
            # Last i elements are already in place 
            for j in range(0, n-i-1): 
                if coor[j+head_start][0] > coor[j+1+head_start][0] :
                    coor[j+head_start] , coor[j+1+head_start] = coor[j+1+head_start] , coor[j+head_start]
                    img[j] , img[j+1] = img[j+1] , img[j]
        return img , coor     	

    def removeRedundant(self , img , coor) :

        
        i = 0
        n = len(img)
        no_redundancy_img = []
        no_redundancy_coor = {}
        
        #for i in range(n-1) :
        #print(n , 'coor redundant' , coor)
        #print('******')
        include = False
        while i < n-1 :
            l_x_min = coor[i][0]
            l_x_max = coor[i][1]
            l_y_min = coor[i][2]
            l_y_max = coor[i][3]
            r_x_min = coor[i+1][0]
            r_x_max = coor[i+1][1]
            r_y_min = coor[i+1][2]
            r_y_max = coor[i+1][3]
            
            #print(i)
            include = False
            if l_x_max > r_x_min :
                if (l_x_max-l_x_min)*(l_y_max-l_y_min) >= (r_x_max-r_x_min)*(r_y_max-r_y_min) :     
                    no_redundancy_img.append(img[i])
                    no_redundancy_coor[i] = coor[i]
                else :
                    no_redundancy_img.append(img[i+1])
                    no_redundancy_coor[i] = coor[i+1]
                i += 1
            else :
                no_redundancy_img.append(img[i])
                no_redundancy_coor[i] = coor[i]
                include = True
            i += 1
        if include :
            no_redundancy_img.append(img[n-1])
            no_redundancy_coor[i] = coor[n-1]
                        

        return no_redundancy_img , no_redundancy_coor
