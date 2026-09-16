% Generate position_order and positions CSVs matching teensyRandomPositionStim output.
% Edit parameters below to match your experiment call, then run once.

positions   = linspace(0, 120, 8);    % must match the positions= passed at the rig;
                                      % NOT the script default (a single position, 50)
n_positions = length(positions);
n_repeats   = 10;                       % same default as teensyRandomPositionStim

position_order = zeros(n_repeats, n_positions);
rng(0);
for rep = 1:n_repeats
    position_order(rep, :) = randperm(n_positions);
end

% position_order: indices (1-based) into positions, shape (n_repeats x n_positions)
% positions:      actual pixel x-values, shape (1 x n_positions)
writematrix(position_order, 'position_order.csv');
writematrix(positions,      'positions.csv');
fprintf('Saved position_order.csv and positions.csv\n');
fprintf('  position_order: %d repeats x %d positions (1-based indices)\n', n_repeats, n_positions);
fprintf('  positions: %s\n', mat2str(round(positions)));
