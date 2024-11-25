function frame_timestamp = triggerCamera(v,w, triggerer, options)

arguments (Input)
    v
    w
    triggerer
    options.TotalFrames (1,1) = 100
    options.ROISelection (1,1) logical = true
    options.FileLocation = "C:\Users\Goard Lab\Documents\GitHub\Distance-Coding-Analysis\hardware-control"
end

% Select ROI
if options.ROISelection
    ROI = extractROI(v);
    ROI_2 = extractROI(w, ROI=ROI);
end

% Start the files where the video is going to log on
logfile_v = startLogfile(FileLocation=options.FileLocation, FileName='camera_1');
logfile_w = startLogfile(FileLocation=options.FileLocation, FileName = 'camera_2');

open(logfile_v)
open(logfile_w)

% Make figure
fig = figure('Units','normalized', 'Position', [0.1,0.2,0.7,0.3]);
ax1 = subplot(1,2,1);
ax2 = subplot(1,2,2);
im1 = image(ax1, snapshot(w));
im2 = image(ax2, snapshot(v));
title(ax1, 'Right Eye', 'FontSize', 14);
title(ax2, 'Left Eye', 'FontSize', 14);
axis([ax1, ax2], 'off', 'image', 'equal')

colormap(ax1, "gray")
colormap(ax2, "gray")

% Triggerer options
configureTerminator(triggerer, "LF");
configureCallback(triggerer, "terminator", @readSerialData)

% Start triggering
frame_timestamp = zeros(1, options.TotalFrames);
is_first_triggered = false;
i_frame = 0;
while i_frame < options.TotalFrames

    is_triggered = readSerialData(triggerer);

    if is_triggered && ~is_first_triggered
        tic;
        is_first_triggered = true;
    end

    if is_triggered
        
        i_frame = i_frame + 1;
        frame_timestamp(i_frame) = toc;
        
        [snap_1, ~] = writeFrame(v, logfile_v, ROI);
        [snap_2, ~] = writeFrame(w, logfile_w, ROI_2);
      
        set(im1, 'CData', snap_1);
        set(im2, 'CData', snap_2);

        sgtitle(fig, "Frame " + string(i_frame) + "/" + string(options.TotalFrames))

    end

end


function is_triggered = readSerialData(src, ~)

data = readline(src);

if strcmp(data, '1')
    is_triggered = 1;
else
    is_triggered = 0;
end

end





























end