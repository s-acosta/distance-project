function simulateScope(options)
% Simulate the scope's frame clock on an NI DAQ analog output, to test radialDeviceControl_v3 with
% trigger "arduino" (startTrigger.ino) without the scope.
%
% Usage (on the DAQ computer; the distance script runs on the other one):
%   simulateScope()                           % 10 Hz until Ctrl-C
%   simulateScope('rate', 30, 'duration', 60)
%
% Wiring: DAQ ao0 -> trigger Arduino D2, DAQ AO GND -> trigger Arduino GND.
%
% Same order as a real session:
%   1. Run radialDeviceControl_v3 on the other computer and press enter there to arm.
%   2. Run this and press enter: the first pulse starts the session there.
% Running it before arming checks that pulses from focus / live view are ignored.
%
% Pulses are software-timed, so they work on any NI device with an analog output but jitter by a
% few ms. Only the first edge matters to the trigger.
%
% Options:
%   device      - NI device. Default: "Dev1"
%   channel     - Analog output. Default: "ao0"
%   rate        - Frames per second. Default: 10
%   pulse_width - Seconds high per frame. Default: 0.01
%   high        - Pulse voltage; keep it at or below the Arduino's 5 V. Default: 5
%   duration    - Seconds of pulses; Ctrl-C stops earlier. Default: inf

arguments
    options.device = "Dev1";
    options.channel = "ao0";
    options.rate (1,1) double {mustBePositive} = 10;
    options.pulse_width (1,1) double {mustBePositive} = 0.01;
    options.high (1,1) double {mustBeInRange(options.high, 0, 5)} = 5;
    options.duration (1,1) double {mustBePositive} = inf;
end

if options.pulse_width >= 1 / options.rate
    error('pulse_width (%g s) must be shorter than one frame (%g s).', options.pulse_width, 1 / options.rate);
end

DAQ = daq('ni');
DAQ.addoutput(options.device, options.channel, 'Voltage');
DAQ.write(0);
cleanup = onCleanup(@() DAQ.write(0));  % back to 0 V on Ctrl-C or error

fprintf('Frame clock on %s/%s: %g Hz, %g ms pulses, 0 -> %g V\n', options.device, options.channel, ...
        options.rate, 1000 * options.pulse_width, options.high);
fprintf('Press enter to start the "acquisition" (Ctrl-C stops)\n');
pause();

n = 0;  % frames sent; frame n + 1 starts at n / rate
t0 = tic;
while n / options.rate < options.duration
    while toc(t0) < n / options.rate
        pause(0.0005);
    end
    DAQ.write(options.high);
    while toc(t0) < n / options.rate + options.pulse_width
        pause(0.0005);
    end
    DAQ.write(0);
    n = n + 1;
    if n == 1
        fprintf('First frame sent\n');
    elseif mod(n, round(10 * options.rate)) == 0
        fprintf('  %d frames (%.0f s)\n', n, toc(t0));
    end
end
fprintf('Stopped after %d frames (%.1f s)\n', n, toc(t0));

end
