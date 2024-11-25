v = startCamera(2);
w = startCamera(1);

previewCamera(v, w);

%%

triggerer = serialport('com10', 115200);
triggerer.Timeout = 1000;
triggerCamera(v, w, triggerer)

%%

ROI = extract_ROI(v);

logfile_1 = startLogfile(FileLocation='C:\Users\Goard Lab\Documents\GitHub\Distance-Coding-Analysis\hardware-control', FileName='camera_1');
logfile_2 = startLogfile(FileLocation = 'C:\Users\Goard Lab\Documents\GitHub\Distance-Coding-Analysis\hardware-control', FileName = 'camera_2');

open(logfile_1);
open(logfile_2);

fig = figure('Units','normalized', 'Position', [0.1,0.2,0.4,0.4]);
ax1 = gca;
im1 = image(ax1, imcrop(rgb2gray(snapshot(v)), ROI));

axis(ax1, 'off', 'image', 'equal')
title(ax1, 'Left Eye', 'FontSize', 12);

trigger_port = 'com9';
triggerer = serialport(trigger_port, 115200);

configureTerminator(triggerer, "LF");
configureCallback(triggerer, "terminator", @readSerialData)

i_frame = 0;
frame_rate = 20;
total_frames = 50 ;
frame_timestamp = zeros(1, total_frames);
timestamps = frame_timestamp;

is_first_triggered = false;
while i_frame < total_frames

    is_triggered = readSerialData(triggerer);

    if is_triggered && ~is_first_triggered
        tic;
        is_first_triggered = true;
    end

    if is_triggered
        
        i_frame = i_frame + 1;
        frame_timestamp(i_frame) = toc;
        
        [snap_1, timestamp] = write_frame(v, logfile_1, ROI);
        timestamps(i_frame) = timestamp;
        
        set(im1, 'CData', snap_1)
        sgtitle(fig, "Frame " + string(i_frame) + "/" + string(total_frames))

    end

end

close(logfile_1)
configureCallback(triggerer, "off" )
clear triggerer

%%
function is_triggered = readSerialData(src, ~)

data = readline(src);

if strcmp(data, '1')
    is_triggered = 1;
else
    is_triggered = 0;
end

end




