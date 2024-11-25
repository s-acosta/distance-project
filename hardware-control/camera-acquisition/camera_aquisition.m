% Configure the device to log to disk using the video writer
total_frames = 0.5*60*10;
frame_rate = 10;

%% Step 1: Access and Configure a Device.
filelocation = "D:\data-distance-coding-project\misc";
filename = "test.mp4";
fullFilename = fullfile(filelocation, filename);

% Create and configure the video writer
logfile = VideoWriter(fullFilename, "MPEG-4");
logfile.FrameRate = 10;
logfile.Quality = 75;

%%
% Access an image acquisition device.
v = videoinput('winvideo', 2);

w = videoinput("winvideo", 3);

v.LogginMode = "disk";
v.DiskLogger = logfile;

v.LoggingMode = "disk";
v.DiskLogger = logfile;

% Set recording parameters
triggerconfig(v, 'manual');
v.FramesPerTrigger = 1;
v.TriggerFrameDelay = 0;
v.TriggerRepeat = inf;

% Preview what the camera is seeing
preview(v);

%% Select ROI
snap = getsnapshot(v);

% display it
f = figure;
ax = gca;
title("Select your ROI and double click when finished:")
imagesc(snap)
axis image
axis off
colormap gray

roi = drawrectangle(ax);
v.ROIPosition = round(customWait(roi)); % get rounded position
close(f);
preview(v); % recall to update ROI
fprintf("ROI updated.\n");

%% Start the object
start(v);

%% Wait for acquisition
pause;

%% Acquire
tic;
number_of_frames = 100;
i_frame = 0;
frame_timestamp = [];

while i_frame < total_frames

    if toc > (i_frame) * 1/frame_rate
        frame_timestamp(end+1) = toc;
        trigger(v);
        i_frame = i_frame + 1;
    end

end
stop(v);

%% Delete object
delete(imaqfind);

%%
function pos = customWait(hROI)

% Listen for mouse clicks on the ROI
l = addlistener(hROI,'ROIClicked',@clickCallback);

% Block program execution
uiwait;

% Remove listener
delete(l);

% Return the current position
pos = hROI.Position;

end

function clickCallback(~,evt)

if strcmp(evt.SelectionType,'double')
    uiresume;
end

end
