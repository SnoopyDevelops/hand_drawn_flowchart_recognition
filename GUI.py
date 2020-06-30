from tkinter import Tk, Label, Button, filedialog, Menu, Toplevel

import cv2

from flowchart_recognition import flowchart


def alert_popup(title, message):
    """Generate a pop-up window for special messages."""
    root = Tk()
    root.title(title)
    w, h = 400, 200  # popup window width and height
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (sw - w) / 2, (sh - h) / 2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message + '\n'
    w = Label(root, text=m, width=50, height=10)
    w.pack()
    b = Button(root, text="OK", command=root.destroy, width=5)
    b.pack()


def database():
    # pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    try:
        filename = root.filename
    except:
        alert_popup('Error message!!!', 'Select a file first')
        return
    if filename == '':
        alert_popup('Error message!!!', 'Select a file first')
        return

    popup = Toplevel()
    Label(popup, text="Processing..").grid(row=0, column=0)

    popup.pack_slaves()
    popup.update()
    flowchart(filename)
    popup.destroy()

    root.filename = ''
    alert_popup('Completed', 'Your task has been completed..')


def OpenFile():
    root.filename = filedialog.askopenfilename(
        title="Select file",
        filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*"))
    )
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.imshow("output", cv2.resize(cv2.imread(root.filename), (2000, 750)))  # Show image
    cv2.waitKey(0)


def About():
    popup = Toplevel()
    Label(popup, text="Recognition of Hand Drawn Flowcharts").grid(row=10, column=1)


def show():
    root.filename = filedialog.askopenfilename(
        title="Select file",
        filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*"))
    )


if __name__ == "__main__":
    root = Tk()
    root.geometry('500x400')
    root.title("hand_drawn_flowchart_recognition")

    menu = Menu(root)
    root.config(menu=menu)

    filemenu = Menu(menu)
    menu.add_cascade(label="File", menu=filemenu)

    filemenu.add_command(label="Open...", command=OpenFile)
    filemenu.add_separator()

    filemenu.add_command(label="Exit", command=root.destroy)

    help_menu = Menu(menu)
    menu.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About...")

    label_0 = Label(root, text="Hand-Drawn Flowchart \n Recognition", width=20, font=("bold", 20))
    label_0.place(x=100, y=50)

    Button(root, text='Select Image', width=30, bg='blue', fg='white', command=show).place(x=150, y=150)
    Button(root, text='Start', width=30, bg='brown', fg='white', command=database).place(x=150, y=250)

    root.mainloop()
