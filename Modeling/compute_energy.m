
function E = compute_energy(m_p, d)

m_d = 1.2;        % drone mass (kg)
r = 0.1;          % prop radius (m)
A_p = pi*r^2;     % single-prop disk area (m^2)
A_f = 0.02;       % frontal area for horizontal drag (m^2)  (estimate)
A_t = 0.02;       % frontal area for vertical drag (m^2)    (estimate)
C_d = 0.8;        % drag coefficient (estimate)
C_t = 0.03;       % thrust coefficient (typical 0.01-0.05)
rho = 1.225;      % air density (kg/m^3)
g = 9.81;         % gravity (m/s^2)

v_h = 7;          % horizontal speed (m/s)
v_v = 3;          % vertical speed (m/s)
h = 40;           % VTOL altitude (m)

w_p = m_d + m_p; % payload weight


w_h = @(w_p, v_h) ((4*(w_p)^2*g^2+rho^2*A_f^2*C_d^2*v_h^4)^(1/4)) ...
      / (r^2*rho*A_p*C_t)^(1/4);

w_v = @(w_p, v_v) ((2*(w_p)*g+rho^2*A_t^2*C_d^2*v_v^4)^(1/4)) ...
      / (r^2*rho*A_p*C_t)^(1/2);

P = @(w) 2.258e-7*w_h^3 + 3.866e-5*w_h^2 + 5.137e-3*w_h + 2.616;

E = w_v(w_p, v_v) * (h/v_v) +...
    w_h(w_p, v_h) * (d/v_h) +...
    w_v(w_p, -v_v) * (h/abs(-v_v)) +...
    w_v(0, v_v) * (h/v_v) +...
    w_h(0, v_h) * (d/v_h) +...
    w_v(0, -v_v)  * (h/abs(-v_v));