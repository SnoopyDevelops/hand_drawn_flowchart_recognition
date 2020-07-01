# HandWritten Flowchart Recognition with Flask Backend Service
 
 Sample Input Image<br/>
![Sample Input Image](photo.png)

- command line: python flowchart_recognition.py "image file name" [parameters]<br/>
If image file name were not given, it'll be asking image file name

- GUI: python GUI.py

- flask backend service: python main.py

Result

| Id | Name | Position | Shape | Line |
| - | - | - | - | - |
| 0 | DATABASE| Inside | square |  |
| 2 | LORD BALANCER | Inside | square | DATABASE |
| 4 | LMN | Inside | square | LORD BALANCER|

* Position: inside / outside
* Shape: triangle / square / circle
* Line: connected nodes to go

parameters:

- padding: the distance range between inside text and boundingRect of shapes
- offset: used to decide direction(AC or BD) of curved lines
- arrow: length of arrow

Output Image for Sample Input Image<br/>
![Output Image](photo_out.png)
