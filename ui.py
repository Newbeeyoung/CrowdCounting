import tkinter as tk
from tkinter.filedialog import  askopenfilename
from PIL import ImageTk,Image

path="a.png"

def openfile():
    path=askopenfilename(initialdir="C:/User",
                              filetypes=(("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")),
                              title="Choose File")
    h = window.winfo_height()
    w = window.winfo_width()

    img_or=Image.open(path)
    img_rs=img_or.resize((int(4*w/9.0),h),Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img_rs)
    image1.configure(image=img_tk)
    image1.Image = img_tk

def estimate():
    path="b.png"

    h = window.winfo_height()
    w = window.winfo_width()

    img_or = Image.open(path)
    img_rs = img_or.resize((int(4*w/9.0),h), Image.ANTIALIAS)
    img_tk1 = ImageTk.PhotoImage(img_rs)
    image2 = tk.Label(window, image=img_tk1)
    image2.Image = img_tk1
    image2.pack()
    image2.place(x=int(5*w/9.0),y=0)


window=tk.Tk()
window.title("Crowd Counting")
window.geometry("1920x1080")
window.configure(background="grey")
# imagefile=ImageFile()
print("Height"+str(window.winfo_height())+"Width"+str(window.winfo_width()))

menu=tk.Menu(window)
window.config(menu=menu)

file=tk.Menu(menu)
file.add_command(label="Open",command=openfile)
file.add_command(label="Exit",command=lambda:exit())
menu.add_cascade(label="File",menu=file)

img_or=Image.open(path)
img_rs=img_or.resize((1,1), Image.ANTIALIAS)
img_tk= ImageTk.PhotoImage(img_rs)
image1= tk.Label(window, image=img_tk)
image1.Image = img_tk
image1.pack()
image1.place(x=0,y=0)

button=tk.Button(window,text="Estimate",command=estimate)
button.pack()
button.place(x=940,y=480)

window.mainloop()

