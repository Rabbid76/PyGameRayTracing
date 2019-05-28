Captured images while rendering are stored in this folder.
The names of the images are "img_0.png", "img_1.png", ...
The time interval for capturing is set in seconds, by the 3rd parameter of `Application.run()`.
0 means that capturing is off.

e.g. 60 seconds:

```py
app.run(render, 100, 60)
```