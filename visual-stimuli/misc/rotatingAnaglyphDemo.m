% rotatingAnaglyphDemo.m
sca; close all; clear;

PsychDefaultSetup(2);
screens      = Screen('Screens');
screenNumber = max(screens);
white        = WhiteIndex(screenNumber);
black        = BlackIndex(screenNumber);
[win, windowRect] = PsychImaging('OpenWindow', screenNumber, black);
[xC, yC]     = RectCenter(windowRect);
Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

% Parameters
winW        = windowRect(3);
winH        = windowRect(4);
radius      = min(winW, winH) * 0.45;
numDots     = 15000;
dotSize     = 2;
disparity   = 100;             % total pixel shift
halfDisp    = disparity / 2;
rotationPeriod = 10;           % seconds per full 360°
rotationSpeed  = 360 / rotationPeriod;  % deg per second

% Generate random‐dot positions in a circle (relative coords)
theta = rand(1, numDots) * 2*pi;
r     = sqrt(rand(1, numDots)) * radius;
x0    =  r .* cos(theta);
y0    =  r .* sin(theta);

tStart = GetSecs;
vbl    = Screen('Flip', win);

try
    while ~KbCheck
        % Compute current rotation angle (0–360)
        angle = mod((GetSecs - tStart) * rotationSpeed, 360);
        
        % Rotate the dot field
        xr = x0 * cosd(angle) - y0 * sind(angle);
        yr = x0 * sind(angle) + y0 * cosd(angle);
        
        % Map to screen space
        xScreen = xr + xC;
        yScreen = yr + yC;
        
        % Prepare dot matrices (2xN)
        xyRed  = [xScreen - halfDisp; yScreen];
        xyCyan = [xScreen + halfDisp; yScreen];
        
        % Draw background as black
        Screen('FillRect', win, black);
        
        % Draw red‐shifted dots
        Screen('DrawDots', win, xyRed, dotSize, [255 0 0], [0 0], 1);
        % Draw cyan‐shifted dots
        Screen('DrawDots', win, xyCyan, dotSize, [0 0 255], [0 0], 1);
        
        % Flip to screen
        vbl = Screen('Flip', win, vbl + 0.5 * Screen('GetFlipInterval', win));
    end
catch
    sca;
    psychrethrow(psychlasterror);
end

sca;
