function alignmentStim(opts)
% Static alignment stimulus for mouse goggles via Teensy.
% Displays two white balls, one on each side, and streams the frame
% continuously until the user presses Ctrl+C.
%
% Usage:
%   alignmentStim()
%   alignmentStim('positions', [60 180], 'stim_size', 40)
%   alignmentStim('preview', true)  % Preview on screen only

arguments
    opts.positions (1,2) double = [120]       % x positions of left/right balls
    opts.stim_size (1,1) double = 30           % ball diameter (pixels)
    opts.rcycles   (1,1) double = 1              % radial checkerboard cycles
    opts.tcycles   (1,1) double = 2              % angular checkerboard cycles
    opts.PORT      (1,1) string = "COM5"
    opts.baud      (1,1) double = 115200
    opts.H         (1,1) double = 240
    opts.W         (1,1) double = 240
    opts.target_fps(1,1) double = 10
    opts.eye       (1,1) string  = "none"        % eye orientation: "none", "left" (+90°), "right" (-90°)
    opts.preview   (1,1) logical = false
    opts.bw        (1,1) logical = false         % black background, white balls
end

%% Build frame with two balls
if opts.bw
    frame = uint8(zeros(opts.H, opts.W, 3));
else
    frame = uint8(ones(opts.H, opts.W, 3) * 127);
end
if opts.stim_size > 0
    for i = 1:length(opts.positions)
        frame = place_ball(frame, opts.positions(i), opts.stim_size, ...
            opts.W, opts.H, opts.rcycles, opts.tcycles, opts.bw);
    end
end
frame = rotate_for_eye(frame, opts.eye);
payload = rgb565_payload(frame);

fprintf('=== Alignment Stimulus ===\n');
fprintf('Ball positions: [%d, %d]\n', opts.positions(1), opts.positions(2));
fprintf('Ball size: %d pixels\n', opts.stim_size);
if opts.bw
    fprintf('Mode: B&W (white balls, black background)\n');
end
if opts.preview
    fprintf('MODE: PREVIEW (screen only)\n');
else
    fprintf('Port: %s\n', opts.PORT);
end
fprintf('Eye orientation: %s\n', opts.eye);
fprintf('==========================\n\n');

%% Send
if opts.preview
    fig = figure('Name', 'Alignment Preview', 'NumberTitle', 'off');
    ax = axes(fig);
    imshow(frame, 'Parent', ax);
    title(ax, 'Alignment');
    drawnow;
else
    s = serialport(opts.PORT, opts.baud, "Timeout", 2);
    configureTerminator(s, "LF");
    flush(s);
    s.OutputBufferSize = 300000;
    writeline(s, sprintf('IMG %d %d', opts.W, opts.H));
    write(s, payload, 'uint8');
end

end

function frame = place_ball(frame, x_pos, stim_size, W, H, rcycles, tcycles, bw)
    xylim = pi * rcycles;
    [xg, yg] = meshgrid(linspace(-xylim, xylim, stim_size));
    circle = (xg.^2 + yg.^2) <= xylim^2;
    if bw
        ball = double(circle);
    else
        at = atan2(yg, xg);
        checksN = (1 + sign(sin(at * tcycles) + eps) .* sign(sin(sqrt(xg.^2 + yg.^2)))) / 2;
        ball = circle .* checksN + 0.5 * ~circle;
    end
    ball_rgb = uint8(round(ball * 255));
    ball_rgb = cat(3, ball_rgb, ball_rgb, ball_rgb);

    y_center = round(H / 2);
    half = floor(stim_size / 2);

    y1 = max(1, y_center - half);
    y2 = min(H, y_center - half + stim_size - 1);
    x1 = max(1, round(x_pos) - half);
    x2 = min(W, round(x_pos) - half + stim_size - 1);

    by1 = y1 - (y_center - half) + 1;
    by2 = by1 + (y2 - y1);
    bx1 = x1 - (round(x_pos) - half) + 1;
    bx2 = bx1 + (x2 - x1);

    frame(y1:y2, x1:x2, :) = ball_rgb(by1:by2, bx1:bx2, :);
end

function frame = rotate_for_eye(frame, eye)
    if strcmp(eye, 'left')
        frame = flip(frame, 2);    % invert x axis: position 0 → rightmost
        frame = rot90(frame, -1);   % +90° CCW
    elseif strcmp(eye, 'right')
        frame = rot90(frame, 1);  % -90° CW
    end
end

function payload = rgb565_payload(img)
    if ~isa(img,'uint8'), img = im2uint8(img); end
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.', [], 1), 'uint8');
end
