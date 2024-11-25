%% Configure camera
% Configure the device to log to disk using the video writer

% Step 1: Access and Configure a Device.

total_frames = 0.5*60*10;
frame_rate = 10;

filelocation = "C:\Users\Goard Lab\Desktop";
filename = "distance-2.mp4";
fullFilename = fullfile(filelocation, filename);

% Create and configure the video writer
logfile = VideoWriter(fullFilename, "MPEG-4");
logfile.FrameRate = 10;
logfile.Quality = 75;

%%

% Access an image acquisition device.
v = videoinput("winvideo", 2, "MJPG_1024x768");

v.LoggingMode = "disk";
v.DiskLogger = logfile;

% Set recording parameters
triggerconfig(v, 'manual');
v.FramesPerTrigger = 1;
v.TriggerFrameDelay = 0;
v.TriggerRepeat = inf;

% Preview what the camera is seeing
preview(v);
start(v);
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

start(v);

%% Position test
% This scripts is to test that the device moves to the desired position
% when prompted with Matlab
n_trials = 1;
start_position = 0;
end_position = 700;
off_time = 0;

stepper_port = 'COM3';
stepper = serialport(stepper_port, 115200);
configureTerminator(stepper, "LF");
while ~contains(stepper.readline(),  "Arduino is ready")
end

stepper.flush();
stepper.UserData = struct("Steps",[],"Time", []);
configureCallback(stepper,"terminator", @readStepsCount)

%% position to go
tic;
stepper.flush();
stepper.UserData = struct("Steps",[],"Time", []);
camera_times = toc;
i_trial = 0;
is_going = 0;
is_coming_back = 0;
stop = 0;

frame_timestamp = [];
i_frame = 0;

trial_position = [end_position, start_position];

while i_trial < 2

    [stepper, stop] = readStepsCount(stepper);
    
    if toc > (i_frame) * 1/frame_rate
        frame_timestamp(end+1) = stepper.UserData.Time(end);
        trigger(v);
        i_frame = i_frame + 1;
    end

    
    if is_going == 0
        order = strcat("<", num2str(trial_position(i_trial+1)), ">");
        stepper.writeline(order);
        is_going = 1;
    end

    if stop == 1
        disp('Stop read')
        i_trial = i_trial + 1;
        is_going = 0;
    end

end


configureCallback(stepper, "off")
data = struct;
data.time = stepper.UserData.Time;
data.steps = stepper.UserData.Steps;
data.frame_timestamp = frame_timestamp;

delete(imaqfind);
plot(data.time./1000, data.steps)

clear stepper

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


