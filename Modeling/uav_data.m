% -----------------------------
% PARAMETERS
% -----------------------------

% Battery parameters

max_joules = 1.6e6;     %assuming 40000mah and 11.1V
num_trials = 10000;       % Monte Carlo iterations

% Preallocate arrays for speed
payloads = 0.2 + (5 - 0.2) * round(rand(num_trials, 1), 2);
distances = 30000 + (100000 - 30000) * round(rand(num_trials, 1), 4);
battery_frac = round(rand(num_trials, 1), 4);
joules_available = battery_frac * max_joules;

% Variability parameters (real-world uncertainty)
wind_std = 0.10;        % ±10% wind effect
efficiency_std = 0.05; % ±5% motor/prop efficiency
sensor_std = 2000;     % ±2000 J measurement noise

% Results storage
energy_used = zeros(num_trials, 1);

% -----------------------------
% SIMULATION LOOP
% -----------------------------
% --- Simulation loop ---
for i = 1:num_trials
    % Base deterministic energy
    E_base = compute_energy(payloads(i), distances(i));

    % Environmental variability
    wind_factor = 1 + wind_std * randn();
    efficiency_factor = 1 + efficiency_std * randn();

    % Apply variability
    E_var = E_base * wind_factor * efficiency_factor;

    % Sensor / measurement noise
    E_measured = E_var + sensor_std * randn();

    % Store final energy (ensure non-negative)
    energy_used(i) = max(round(E_measured, 0), 0);
end


% --- Create table and write to CSV ---
results = table(payloads, distances, joules_available, energy_used);
writetable(results, 'cont_drone_simulation_data.csv');