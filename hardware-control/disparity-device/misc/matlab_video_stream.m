%% teensy_stream_video.m — send each video frame as one full RGB565 buffer
PORT  = "COM7";          % Windows: "COMx", macOS: "/dev/tty.usbmodemXXXX", Linux: "/dev/ttyACM0"
H = 240; W = 240;        % your panel size

% --- Open serial ---
s = serialport(PORT, 115200, "Timeout", 2);
configureTerminator(s, "LF");
flush(s);
s.OutputBufferSize = 300000;   % > 115,200 bytes per frame

%%

% --- Pick a sample video (change if needed) ---
VIDEO = "stim.mp4";
WIDEO = "xylophone.mp4";
% try
%     VIDEO = fullfile(matlabroot,'toolbox','matlab','audiovideo','xylophone.mp4');
%     assert(isfile(VIDEO));
% catch
% 
% VIDEO = "xylophone.mp4";   % or any local file
% end

v = VideoReader(VIDEO);
w = VideoReader(WIDEO);

% Cap send rate to something reasonable
targetFPS = 20;
frameInterval = 1/targetFPS;

% --- Helper: RGB uint8 -> RGB565 bytes (row-major, LITTLE-endian) ---
toPayload = @(img) ...
    typecast( reshape( rgb565_from_rgb(img).', [], 1 ), 'uint8' );
% (transpose to W×H for row-major order, then reshape to vector before typecast)

% --- Stream loop ---
fprintf('Streaming %s at up to %.1f fps...\n', VIDEO, targetFPS);
lastSent = tic; 
frames = 0; 
t0 = tic;
t00 = tic;

while hasFrame(v)
    rgb = readFrame(v);                 % HxWx{3 or 1}, typically double in [0,1]
    if ~isa(rgb,'uint8'), rgb = im2uint8(rgb); end
    if ndims(rgb)==2, rgb = repmat(rgb,[1 1 3]); end
    rgb = imresize(rgb, [H W]);         % match panel

    payload = toPayload(rgb);           % row-major LE bytes

    % Send header and payload (Teensy full-frame sketch expects this)
    writeline(s, sprintf('IMG %d %d', W, H));
    write(s, payload, "uint8");

    % % simple pacing
    % dt = toc(lastSent);
    % while dt < frameInterval
    % end

    lastSent = tic;

    % (optional) FPS print
    frames = frames + 1;
    if frames == 60
        elapsed = toc(t0); fprintf('~%.1f fps\n', frames/elapsed);
        frames = 0; t0 = tic;
    end
end

time = tic - t00;
fprintf("Done.\n");

% --- helper function (kept local for convenience) ---
function rgb565 = rgb565_from_rgb(img)
    % img: HxWx3 uint8
    R5 = bitshift(uint16(img(:,:,1)), -3);
    G6 = bitshift(uint16(img(:,:,2)), -2);
    B5 = bitshift(uint16(img(:,:,3)), -3);
    rgb565 = bitor( bitshift(R5,11), bitor(bitshift(G6,5), B5) ); % HxW uint16
end
