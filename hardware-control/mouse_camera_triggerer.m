v = startCamera(1, Resolution='1600x1200');
previewCamera(v);

triggerer = arduino("COM10", 'Uno');
frame_timestamp = triggerCameraDAQ(v, triggerer, TotalFrames=1800, FileName = '241120_test_2');