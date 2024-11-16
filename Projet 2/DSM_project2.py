import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

ACC_REF = 10**-6

############ IRF analysis ############

# Load the data from irf text file
irf_data = np.loadtxt('irf_acc.txt')

# Extract time steps and acceleration values
irf_time = irf_data[:, 0]
irf_acc = irf_data[:, 1]

irf_time_step = (irf_time[-1] - irf_time[0])/(len(irf_time)-1)

# Remove time <= (0.02 + ...) to avoid aberrations due to the pothole
pidx = 0
while irf_time[pidx] < 0.02:
    pidx += 1

pothole_index = pidx + 10

irf_filtered_time = irf_time[pothole_index:]
irf_filtered_acc = irf_acc[pothole_index:]


# peaks search, natural frequency & damping ratio
peaks, _ = find_peaks(irf_filtered_acc)
peaks = peaks[:5]  # take 5 first peaks, becomes too unsignificant after
peaks_time = irf_filtered_time[peaks]
peaks_acc = irf_filtered_acc[peaks]

nat_freq_irf = ((peaks_time[-1] - peaks_time[0]) /
                (len(peaks)-1))**(-1)    # divide by n-1 because
# there are n-1 intervals
# between n peaks
log_dec = np.log(peaks_acc[0]/peaks_acc[-1])/(len(peaks)-1)

damping_ratio_irf_approx = log_dec/(2*np.pi)
damping_ratio_irf_exact = log_dec/(np.sqrt(4 * np.pi**2 + log_dec**2))


############ FRF analysis ############


# Load the data from frf text file
frf_data = np.loadtxt('frf_acc.txt')

# Extract time steps and acceleration values
frf_frequency = frf_data[1:, 0]
frf_real = frf_data[1:, 1] / (-np.power(frf_frequency, 2))
frf_im = frf_data[1:, 2] / (-np.power(frf_frequency, 2))
frf_real, frf_im, frf_frequency = np.insert(frf_real, 0, 0), np.insert(
    frf_im, 0, 0), np.insert(frf_frequency, 0, 0)
# frf_real[0], frf_im[0] = 0,0

### Bode plots (amplitude and phase) and search of nat freq, quali factor and damp ratio ###
bode_phase = -np.arctan2(frf_im, frf_real)


frf_amplitude = np.sqrt(np.power(frf_real[1:], 2) + np.power(frf_im[1:], 2))
bode_amplitude = frf_amplitude  # 20*np.log10(frf_amplitude/ACC_REF)

peak_amplitude_idx, _ = find_peaks(frf_amplitude)
nat_freq_bode = frf_frequency[peak_amplitude_idx+1]

# quality factor computation

bode_amplitude_max = frf_amplitude[peak_amplitude_idx]
bode_halfPower_amplitude = bode_amplitude_max/np.sqrt(2)
bode_halfPower_idx = np.where(
    np.diff(np.sign(frf_amplitude - bode_halfPower_amplitude)) != 0)[0]

halfP_f1, halfP_f2 = frf_frequency[bode_halfPower_idx[0]
                                   ], frf_frequency[bode_halfPower_idx[0]+1]
halfP_a1, halfP_a2 = frf_amplitude[bode_halfPower_idx[0]
                                   ], frf_amplitude[bode_halfPower_idx[0]+1]
halfPower_freq1 = halfP_f1 + \
    (bode_halfPower_amplitude - halfP_a1) * \
    (halfP_f2 - halfP_f1)/(halfP_a2-halfP_a1)

halfP_f3, halfP_f4 = frf_frequency[bode_halfPower_idx[1]
                                   ], frf_frequency[bode_halfPower_idx[1]+1]
halfP_a3, halfP_a4 = frf_amplitude[bode_halfPower_idx[1]
                                   ], frf_amplitude[bode_halfPower_idx[1]+1]
halfPower_freq2 = halfP_f3 + \
    (bode_halfPower_amplitude - halfP_a3) * \
    (halfP_f4 - halfP_f3)/(halfP_a4-halfP_a3)

quality_factor = ((halfPower_freq2 - halfPower_freq1)/nat_freq_bode)**(-1)
bode_damping_ratio = 1/(2*quality_factor)
bode_damping_ratio_prior = bode_damping_ratio

nat_freq_bode_undamped = nat_freq_bode  # 1st guess is the damped freq

for i in range(10):
    nat_freq_bode_undamped = nat_freq_bode/np.sqrt(1-bode_damping_ratio**2)
    quality_factor = ((halfPower_freq2 - halfPower_freq1) /
                      nat_freq_bode_undamped)**(-1)
    bode_damping_ratio = 1/(2*quality_factor)


### Nyquist plot and search of damping ratio ###

# nyquist = M + iN  M = k*Re(H), N = k*Im(H) => k=?
nyquist_M = frf_real/frf_amplitude[0]
nyquist_N = frf_im/frf_amplitude[0]

# zero search in real part (with interpolation), zero happens at undamped nat freq
nyquist_ω0_idx = np.where((np.diff(np.sign(frf_real))) != 0)[
    0][1]  # zero n°1 is 1st point

# Linear interpolation to estimate the frequency where real part crosses zero
f1, f2 = frf_frequency[nyquist_ω0_idx], frf_frequency[nyquist_ω0_idx + 1]
r1, r2 = nyquist_M[nyquist_ω0_idx], nyquist_M[nyquist_ω0_idx + 1]

nyquist_ω0_freq = f1 - r1 * (f2 - f1) / (r2 - r1)

# Now interpolate the imaginary part at the zero-crossing frequency
i1, i2 = nyquist_N[nyquist_ω0_idx], nyquist_N[nyquist_ω0_idx + 1]
nyquist_ω0_im = i1 + (nyquist_ω0_freq - f1) * (i2 - i1) / (f2 - f1)

damping_ratio_nyquist = 1/(2*nyquist_ω0_im)


################################################
#################### PLOTS #####################
################################################

########################
##### IRF ANALYSIS #####
########################

fig, ax = plt.subplots(figsize=(7, 4 * 7/8))

# Plot the main function (IRF acceleration data)
ax.plot(irf_time, irf_acc, color='blue')

# Vertical line at the pothole index and horizontal in y=0
ax.axvline(x=irf_time[pothole_index], color='red', label='Temps du filtrage')
ax.axhline(y=0, color='black', zorder=0)

# Scatter plot for the peaks
ax.scatter(peaks_time, peaks_acc, color='red', s=15, zorder=3,
           label='5 1$^{ers}$ Pics d\'amplitude', marker='x')

# Shade the region to the left of the pothole time
ax.axvspan(0, irf_time[pothole_index], color='grey', alpha=0.2)

# Improve axes appearance
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_position(('outward', -10))  # Offset left spine
ax.spines['bottom'].set_position(('outward', 0))  # Offset bottom spine

# Axes names
ax.set_xlabel('t (s)')
ax.set_ylabel('a (m/s²)')

# Add grid with transparency
plt.grid(True, alpha=0.7)

# Add a title
plt.title(' Fonction de réponse impulsionnelle en accélération')

# Display the legend
plt.legend(loc='upper right')

# Show the plot
fig.savefig('Plots/irf_acceleration_plot.png', dpi=300, bbox_inches='tight')
plt.show()


########################
#### BODE AMPLITUDE ####
########################

fig, ax = plt.subplots(figsize=(7, 7 * 5/8))

# Plot the Bode amplitude
ax.plot(frf_frequency[1:], bode_amplitude,
        color='blue')

ax.scatter(nat_freq_bode, bode_amplitude_max, color='red', 
           label = "maximum d'amplitude",s=20, marker = 'o', zorder = 3)

# Improve axes appearance (as in the style you shared)
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_position(('outward', 10))  # Offset left spine
ax.spines['bottom'].set_position(('outward', 10))  # Offset bottom spine
ax.grid(True, alpha=0.7)  # Add grid for better visualization

# Set the limits for x and y axes if needed (you can customize these)
plt.xlim(0, 1.01*frf_frequency[1:].max())
plt.ylim(0, 1.1*bode_amplitude[1:].max())

# Axes names
ax.set_xlabel('ω(Hz)')
ax.set_ylabel('|H| (m/N)')

# Add a title and legend if needed
ax.set_title('Diagramme de bode en amplitude')

plt.legend(loc = 'lower left')

# Show the plot
fig.savefig('Plots/bode_amplitude.png', dpi=300, bbox_inches='tight')
plt.show()


########################
###### BODE PHASE ######
########################



# Create the figure and axes
fig, ax = plt.subplots(figsize=(7, 5 * 7/8))

# Plot the Bode phase
ax.plot(frf_frequency[1:], bode_phase[1:], color='blue', label='Bode Phase')

# Improve axes appearance
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_position(('outward', 10))  # Offset left spine
ax.spines['bottom'].set_position(('outward', 10))  # Offset bottom spine
ax.grid(True, alpha=0.7)  # Add grid for better visualization

# Set axis limits (customize if needed)
plt.xlim(frf_frequency[1:].min(), frf_frequency[1:].max())
plt.ylim(bode_phase[1:].min(), bode_phase[1:].max())

# Set the standard axis labels
ax.set_xlabel('ω (Hz)')
ax.set_ylabel('φ (rad)')

# Add a title and legend if needed
ax.set_title('Diagramme de Bode en phase')

# Show the plot
fig.savefig('Plots/bode_phase.png', dpi=300, bbox_inches='tight')
plt.show()


########################
####### NYQUIST ########
########################

fig, ax = plt.subplots(figsize=(4.2, 4))

ax.plot(nyquist_M[1:], nyquist_N[1:], color='blue')

# Highlight special points with scatter
ax.scatter(0, nyquist_ω0_im, color='red', zorder=3,s=20, label='fréquence ω₀')

# Add x-axis and y-axis lines
ax.axhline(0, color='black', linewidth=1, zorder=0)  # x-axis
ax.axvline(0, color='black', linewidth=1, zorder=0)  # y-axis

# Improve axes appearance (same as before)
ax.spines['top'].set_visible(False)  # Remove top spine
ax.spines['right'].set_visible(False)  # Remove right spine
ax.spines['left'].set_position(('outward', 10))  # Offset left spine
ax.spines['bottom'].set_position(('outward', 10))  # Offset bottom spine
ax.grid(True, alpha=0.7)  # Add grid for better visualization

# Set axis labels and title
ax.set_xlabel('M (-)')
ax.set_ylabel('N (-)')
ax.set_title('Diagramme de Nyquist')

plt.xlim(1.1*nyquist_M.min(), 1.3*nyquist_M.max())
plt.ylim( 1.1*nyquist_N.min(), -0.2*nyquist_N.min())

# Add legend for better understanding of points
ax.legend(loc='upper left')

# Show the plot
fig.savefig('Plots/nyquist.png', dpi=300, bbox_inches='tight')
plt.show()
