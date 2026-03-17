function stim_log = teensy_play_orientations_common_fixedon_log(orientations, opts)
% Plays: PRE (seconds) -> ON (once) -> POST (seconds) for each orientation,
% using shared PRE/POST videos and per-ori ON videos. Logs times like driftingGratingsStim.
%
% Required files in opts.outdir:
%   <basename>_pre.mp4
%   <basename>_post.mp4
%   <basename>_ori%03d_on.mp4
%
% Returns:
%   stim_log.parameters, stim_log.orientation_sequence
%   stim_log.pre_time_start(i), stim_log.on_time_start(i),
%   stim_log.on_time_end(i),   stim_log.post_time_end(i), finish_time

arguments
    orientations (1,:) double
    opts.PORT (1,1) string = "COM3"
    opts.baud (1,1) double = 115200
    opts.H (1,1) double = 240
    opts.W (1,1) double = 240
    opts.outdir (1,1) string = "C:\Users\acostamendoza\projects\distance-project\hardware-control\disparity-device\stimulus"
    opts.basename (1,1) string = "drift"
    opts.resize (1,1) logical = true
    opts.throttleToVideoFPS (1,1) logical = true
    opts.verbose (1,1) logical = true
    opts.preSeconds = 1
    opts.postSeconds = 1
end

H = opts.H; W = opts.W;
N = numel(orientations);

% ---- Expand durations (assert early) ----
preSecs  = expand_to_vec(opts.preSeconds,  N, "preSeconds");
postSecs = expand_to_vec(opts.postSeconds, N, "postSeconds");

% ---- Build paths (assert early) ----
pre_path  = fullfile(opts.outdir, sprintf("%s_pre.mp4",  opts.basename));
post_path = fullfile(opts.outdir, sprintf("%s_post.mp4", opts.basename));
on_paths  = strings(1,N);
ori3s     = strings(1,N);
for i = 1:N
    ori3s(i) = sprintf('%03d', mod(round(orientations(i)),360));
    on_paths(i) = fullfile(opts.outdir, sprintf("%s_ori%s.mp4", opts.basename, ori3s(i)));
end

if any(preSecs>0),  assert(isfile(pre_path),  "Missing PRE:  %s", pre_path);  end
if any(postSecs>0), assert(isfile(post_path), "Missing POST: %s", post_path); end
for i = 1:N, assert(isfile(on_paths(i)), "Missing ON for orientation[%d]=%g: %s", i, orientations(i), on_paths(i)); end

% ---- Validate clips once (codec + size) ----
if any(preSecs>0),  validate_clip(pre_path,  H, W, opts.resize, "PRE");  end
if any(postSecs>0), validate_clip(post_path, H, W, opts.resize, "POST"); end
for i = 1:N, validate_clip(on_paths(i), H, W, opts.resize, sprintf("ON[%d]",i)); end

% ---- Pre-compute trial durations for summary (like driftingGratingsStim) ----
% We’ll approximate each trial length as preSecs + ON clip duration + postSecs.
on_durs = zeros(1,N);
for i = 1:N
    vr = VideoReader(on_paths(i));
    on_durs(i) = max(0, vr.Duration);
end
trial_durs = preSecs + on_durs + postSecs;
total_trials = N;
fprintf('Total number of trials: %d\n', total_trials);
if all(abs(trial_durs - trial_durs(1)) < 1e-6)
    fprintf('Each trial will take %.2f seconds\n', trial_durs(1));
else
    fprintf('Trial durations vary; mean = %.2f s (min=%.2f, max=%.2f)\n', mean(trial_durs), min(trial_durs), max(trial_durs));
end
fprintf('Total time for all trials: %.2f seconds\n', sum(trial_durs));
fprintf('Total time for all trials: %.2f minutes\n', sum(trial_durs)/60);

% ---- Serial (open once) ----
s = serialport(opts.PORT, opts.baud, "Timeout", 2);
configureTerminator(s, "LF"); flush(s);
try s.OutputBufferSize = max(300000, W*H*2 + 16384); catch, end
if opts.verbose
    fprintf('Serial: %s @ %g baud | Panel: %dx%d | %d orientations\n', opts.PORT, opts.baud, W, H, N);
end

% ---- Prepare log ----
stim_log = struct();
stim_log.parameters = opts;
stim_log.orientation_sequence = orientations(:).';
stim_log.pre_time_start = zeros(1,N);
stim_log.on_time_start  = zeros(1,N);
stim_log.on_time_end    = zeros(1,N);
stim_log.post_time_end  = zeros(1,N);

% ---- Run (tight loop; timing via single tic) ----
start_time = tic;

for i = 1:N
    % PRE
    stim_log.pre_time_start(i) = toc(start_time);
    if preSecs(i) > 0
        if opts.verbose
            fprintf('\n[ORI %s] PRE  %.3fs  (%s)\n', ori3s(i), preSecs(i), fileOnly(pre_path));
        end
        play_video_for_seconds(s, pre_path, preSecs(i), H, W, opts.resize, opts.throttleToVideoFPS, opts.verbose);
    end

    % ON (once)
    stim_log.on_time_start(i) = toc(start_time);
    if opts.verbose
        fprintf('[ORI %s]  ON  (once)  %s\n', ori3s(i), fileOnly(on_paths(i)));
    end
    play_video_once(s, on_paths(i), H, W, opts.resize, opts.throttleToVideoFPS, opts.verbose);
    stim_log.on_time_end(i) = toc(start_time);

    % POST
    if postSecs(i) > 0
        if opts.verbose
            fprintf('[ORI %s] POST %.3fs  (%s)\n', ori3s(i), postSecs(i), fileOnly(post_path));
        end
        play_video_for_seconds(s, post_path, postSecs(i), H, W, opts.resize, opts.throttleToVideoFPS, opts.verbose);
    end
    stim_log.post_time_end(i) = toc(start_time);
end

stim_log.finish_time = toc(start_time);
if opts.verbose, fprintf("\nFinished in %.3f s\n", stim_log.finish_time); end
end

% ================= helpers =================

function vec = expand_to_vec(x, N, name)
    if isscalar(x), vec = repmat(x,1,N);
    else
        assert(numel(x)==N, "%s must be scalar or length %d.", name, N);
        vec = x;
    end
end

function validate_clip(p, H, W, allowResize, label)
    try
        vr = VideoReader(p);
        f = readFrame(vr); %#ok<NASGU>
        if ~allowResize
            assert(vr.Height==H && vr.Width==W, ...
                "%s clip %dx%d must match panel %dx%d (set opts.resize=true to allow resize).", ...
                label, vr.Height, vr.Width, H, W);
        end
    catch ME
        error("Failed to read %s clip '%s': %s", label, p, ME.message);
    end
end

function nm = fileOnly(p)
    [~,nm,ext] = fileparts(p);
    nm = nm+ext;
end

function payload = rgb565_payload(imgRGB)
    if ~isa(imgRGB,'uint8'), imgRGB = im2uint8(imgRGB); end
    R5 = bitshift(uint16(imgRGB(:,:,1)),-3);
    G6 = bitshift(uint16(imgRGB(:,:,2)),-2);
    B5 = bitshift(uint16(imgRGB(:,:,3)),-3);
    rgb565 = bitor(bitshift(R5,11), bitor(bitshift(G6,5), B5));
    payload = typecast(reshape(rgb565.',[],1),'uint8'); % row-major LE
end

function play_video_once(s, videoPath, H, W, doResize, throttle, verbose)
    vr = VideoReader(videoPath);
    if verbose
        fprintf('  > %s | %dx%d @ %.3f fps (%g frames)\n', ...
            videoPath, vr.Height, vr.Width, vr.FrameRate, floor(vr.Duration * vr.FrameRate));
    end
    if throttle && vr.FrameRate>0, frameInterval = 1/vr.FrameRate; else, frameInterval = 0; end
    lastSent = tic; frames = 0; t0 = tic;

    while hasFrame(vr)
        rgb = readFrame(vr);
        if doResize && (size(rgb,1) ~= H || size(rgb,2) ~= W), rgb = imresize(rgb, [H W], "nearest"); end
        payload = rgb565_payload(rgb);
        writeline(s, sprintf('IMG %d %d', W, H));
        write(s, payload, 'uint8');

        if frameInterval > 0
            dt = toc(lastSent); if dt < frameInterval, pause(frameInterval - dt); end
            lastSent = tic;
        end

        frames = frames + 1;
        if verbose && mod(frames, 120) == 0
            elapsed = toc(t0);
            fprintf('    sent %d frames (%.2f fps)\n', frames, frames/max(1e-9,elapsed));
        end
    end
end

function play_video_for_seconds(s, videoPath, seconds, H, W, doResize, throttle, verbose)
% Loop/trim to 'seconds' for PRE/POST (ON never uses this).
    if seconds <= 0, return; end
    target = seconds; sent = 0;
    vr = VideoReader(videoPath);  % open once per segment play
    clipFrames = max(1, floor(vr.Duration * vr.FrameRate));
    frameInterval = (throttle && vr.FrameRate>0) * (1/vr.FrameRate);
    if verbose
        fprintf('  > %s | %dx%d @ %.3f fps (%d frames) — target %.3fs\n', ...
            videoPath, vr.Height, vr.Width, vr.FrameRate, clipFrames, target);
    end

    while sent < target
        remaining = target - sent;
        maxFramesThisPass = clipFrames;
        if vr.FrameRate > 0 && remaining < vr.Duration
            maxFramesThisPass = max(1, floor(remaining * vr.FrameRate));
        end

        lastSent = tic; frames = 0; t0 = tic;
        vr.CurrentTime =  0;
        while hasFrame(vr) && frames < maxFramesThisPass
            rgb = readFrame(vr);
            if doResize && (size(rgb,1) ~= H || size(rgb,2) ~= W), rgb = imresize(rgb, [H W], "nearest"); end
            payload = rgb565_payload(rgb);
            writeline(s, sprintf('IMG %d %d', W, H));
            write(s, payload, 'uint8');

            if frameInterval
                dt = toc(lastSent); if dt < frameInterval, pause(frameInterval - dt); end
                lastSent = tic;
            end

            frames = frames + 1;
            if verbose && mod(frames, 120) == 0
                elapsed = toc(t0);
                fprintf('    sent %d frames (%.2f fps)\n', frames, frames/max(1e-9,elapsed));
            end
        end

        if vr.FrameRate > 0, sent = sent + frames/vr.FrameRate;
        else,                sent = sent + (frames/max(1,clipFrames))*max(vr.Duration,0);
        end
    end
end
