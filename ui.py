import tkinter as tk
from tkinter.filedialog import  askopenfilename
from PIL import ImageTk,Image
from single_img_estimate import single_img_estimate
from src.timer import Timer
import cv2
import numpy as np
import matplotlib as plt

t=Timer()

class UI:
    def __init__(self,master):
        master.title("Crowd Counting")
        master.geometry('1920x1080')
        # master.configure(background='grey')

        menu=tk.Menu(master)
        master.config(menu=menu)
        file=tk.Menu(menu)
        file.add_command(label="Open",command=self.openfile)
        file.add_command(label="Exit",command=lambda:exit())
        menu.add_cascade(label="File",menu=file)

        self.input_path=''
        self.button=tk.Button(master,text="Estimate",command=self.estimate)
        self.button.pack()
        self.button.place(x=890,y=480)
        # Define image size

        self.w=None
        self.h=None
        self.output_x=None


    def openfile(self):

        w_height = window.winfo_height()
        w_width = window.winfo_width()
        print(str(w_width)+";"+str(w_height))
        self.w=int(4*w_width/9.0)
        self.h=int(2*w_height/5.0)

        self.output_x=int(5*w_width/9.0)

        self.input_path=askopenfilename(initialdir="C:/User",
                                  filetypes=(("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")),
                                  title="Choose File")

        #Show original image
        img_or=Image.open(self.input_path)
        img_rs=img_or.resize((self.w,self.h),Image.ANTIALIAS)
        img_tk= ImageTk.PhotoImage(img_rs)
        self.in_image= tk.Label(window, image=img_tk)
        self.in_image.Image = img_tk
        self.in_image.pack()
        self.in_image.place(x=0,y=100)
        #Show name of original image
        var5=tk.StringVar()
        ori_name=tk.Label(window,textvariable=var5,height=2,width=20,font=8)
        var5.set("[Orginal Image]")
        ori_name.place(x=20,y=30)

    def estimate(self):
        #Call estimate function
        et_path,gt_path,mae,mse,gt_count,et_count=single_img_estimate(self.input_path)

        #Show Combined Image
        combined_img=self.compared_image(self.input_path,et_path)
        combined_img=cv2.cvtColor(combined_img,cv2.COLOR_BGR2RGB)
        img_comb=Image.fromarray(combined_img)
        img_comb=img_comb.resize((self.w,self.h),Image.ANTIALIAS)
        img_tk3=ImageTk.PhotoImage(img_comb)
        comb_image=tk.Label(window,image=img_tk3)
        comb_image.Image=img_tk3
        comb_image.pack()
        comb_image.place(x=0,y=(200+self.h))
        # Show estimate image
        print(self.input_path.split('/')[-1].replace('.jpg', '.csv'))
        img_et = Image.open(et_path)
        img_et = img_et.resize((self.w,self.h), Image.ANTIALIAS)
        img_tk1 = ImageTk.PhotoImage(img_et)
        et_image = tk.Label(window, image=img_tk1)
        et_image.Image = img_tk1
        et_image.pack()
        et_image.place(x=self.output_x,y=(200+self.h))

        #Show ground truth image
        img_gt = Image.open(gt_path)
        img_gt =img_gt.resize((self.w,self.h),Image.ANTIALIAS)
        img_tk2 = ImageTk.PhotoImage(img_gt)
        gt_image=tk.Label(window,image=img_tk2)
        gt_image.Image=img_tk2
        gt_image.pack()
        gt_image.place(x=self.output_x,y=100)

        #Show name of gt_heatmap,et_heatmap,combined image
        var6=tk.StringVar()
        gt_name=tk.Label(window,textvariable=var6,height=2,width=20,font=8)
        var6.set("[Ground Truth Heatmap]")
        gt_name.place(x=1100,y=30)

        var7=tk.StringVar()
        et_name=tk.Label(window,textvariable=var7,height=2,width=20,font=8)
        var7.set("[Estimate Heatmap]")
        et_name.place(x=1100,y=560)

        var8=tk.StringVar()
        combined_name=tk.Label(window,textvariable=var8,height=2,width=20,font=8)
        var8.set("[Combined Image]")
        combined_name.place(x=20,y=560)

        #Show mae, mse, gt_count,et_count
        var1=tk.StringVar()
        MAE=tk.Label(window,textvariable=var1,height=2,width=20,font=6)
        var1.set("MAE:"+str("%.2f" %mae))
        MAE.place(x=1400,y=590)

        var2=tk.StringVar()
        MSE=tk.Label(window,textvariable=var2,height=2,width=20,font=6)
        var2.set("MSE:"+str("%.2f" %mse))
        MSE.place(x=1700,y=590)

        var3=tk.StringVar()
        gt_count_txt=tk.Label(window,textvariable=var3,height=2,width=20,font=6)
        var3.set("GT_Count:"+str("%.2f" %gt_count))
        gt_count_txt.place(x=1100,y=60)

        var4=tk.StringVar()
        et_count_txt=tk.Label(window,textvariable=var4,height=2,width=20,font=6)
        var4.set("ET_Count:"+str("%.2f" %et_count))
        et_count_txt.place(x=1100,y=590)

    def compared_image(self,input,estimate):
        input_image=cv2.imread(input)
        estimate_dm=cv2.imread(estimate)
        estimate_dm=cv2.resize(estimate_dm,(input_image.shape[1],input_image.shape[0]))

        combined_image=cv2.addWeighted(input_image,0.4,estimate_dm,0.6,0)
        return combined_image


window=tk.Tk()
b=UI(window)
window.mainloop()

