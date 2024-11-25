function frame_timestamp = triggerCameraDAQ(v, triggerer, options)

arguments (Input)
    v
    triggerer
    options.TotalFrames (1,1) = 100
    options.ArduinoPin = "D2"
    options.ROISelection (1,1) logical = true
    options.FileLocation = "C:\Users\Goard Lab\Documents\GitHub\Distance-Coding-Analysis\hardware-control"
    options.FileName = "test"
end

% Select ROI
if options.ROISelection
    ROI = extractROI(v);
end

% Start the files where the video is going to log on
logfile_v = startLogfile(FileLocation=options.FileLocation, FileName=options.FileName);
open(logfile_v)

% Make figure
fig = figure('Units','normalized', 'Position', [0.1,0.2,0.4,0.5]);
ax1 = gca;
im1 = image(ax1, snapshot(v));
title(ax1, 'Mouse Camera', 'FontSize', 14);
axis(ax1, 'off', 'image', 'equal')
colormap(ax1, "gray")

% Start triggering
frame_timestamp = zeros(1, options.TotalFrames);
i_frame = 1;

trigger_value = 0;
disp('waiting for trigger...');
while ~trigger_value
    trigger_value = triggerer.readDigitalPin(options.ArduinoPin);
end

disp('has been triggered')
tic;

while trigger_value

    if toc > frame_timestamp(i_frame) + 0.1

        i_frame = i_frame + 1;
        frame_timestamp(i_frame) = toc;

        [snap_1, ~] = writeFrame(v, logfile_v, ROI);

        set(im1, 'CData', snap_1);
        sgtitle(fig, "Frame " + string(i_frame) + "/" + string(options.TotalFrames))

    end

    trigger_value =  triggerer.readDigitalPin(options.ArduinoPin);

end


end
