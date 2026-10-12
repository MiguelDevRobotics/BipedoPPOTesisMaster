from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/seed_4000_1000")
ea.Reload()
plt.figure(figsize=(13, 7))
print(ea.Tags()["scalars"])
scale = 1.5

plt.rcParams.update({
    'font.size': 10 * scale,
    'axes.titlesize': 12 * scale,
    'axes.labelsize': 10 * scale,
    'xtick.labelsize': 9 * scale,
    'ytick.labelsize': 9 * scale,
    'legend.fontsize': 10 * scale,
    'lines.linewidth': 1.5 * scale,
})


xy_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_xy")
steps = [x.step for x in xy_vel_error]
values = [x.value for x in xy_vel_error]
#plt.plot(steps, values, label='Error de velocidad lineal 4000 envs')

yaw_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_yaw")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
plt.plot(steps, values, label='Error velocidad angular 4000 envs')

yaw_vel_error = ea.Scalars("Metrics/distancia_recorrida")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
#plt.plot(steps, values, label='Distance traveled 4000 envs')

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/seed_1000_1000")
ea.Reload()
xy_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_xy")
steps = [x.step for x in xy_vel_error]
values = [x.value for x in xy_vel_error]
#plt.plot(steps, values, label='Error de velocidad lineal 1000 envs')

yaw_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_yaw")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
plt.plot(steps, values, label='Error velocidad angular 1000 envs')

yaw_vel_error = ea.Scalars("Metrics/distancia_recorrida")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
#plt.plot(steps, values, label='Distance traveled 1000 envs')

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/seed_500_1000")
ea.Reload()


xy_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_xy")
steps = [x.step for x in xy_vel_error]
values = [x.value for x in xy_vel_error]
#plt.plot(steps, values, label='Error de velocidad lienal 500 envs')

yaw_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_yaw")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
plt.plot(steps, values, label='Error velocidad angular 500 envs')

yaw_vel_error = ea.Scalars("Metrics/distancia_recorrida")
steps = [x.step for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
#plt.plot(steps, values, label='Distance traveled 500 envs')

plt.legend()
plt.tight_layout()
#plt.legend()
plt.title("Errores de seguimiento del comando de velocidad angular")
plt.xlabel("Iteración")
plt.ylabel("Error (rad/s)")
plt.grid()
plt.show()

plt.figure(figsize=(13, 7))


ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-11_20-21-34")
ea.Reload()
feet_slide = ea.Scalars("Episode_Reward/feet_silde")
steps = [x.step for x in feet_slide]
values = [x.value for x in feet_slide]
#plt.plot(steps, values, label='Feet slide 4000 envs')

zmp = ea.Scalars("Episode_Reward/zmp_cop_error")
steps = [x.step for x in zmp]
values = [x.value for x in zmp]
#plt.plot(steps, values, label='Recompensa ZMP 4000 envs')

track_xy = ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")
steps = [x.step for x in track_xy]
values = [x.value for x in track_xy]
#plt.plot(steps, values, label='Velocidad lineal para 4000 envs')

track_yaw = ea.Scalars("Episode_Reward/track_ang_vel_z_exp")
steps = [x.step for x in track_yaw]
values = [x.value for x in track_yaw]
plt.plot(steps, values, label='Velocidad angular para 4000 envs')

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-12_16-04-34")
ea.Reload()
feet_slide = ea.Scalars("Episode_Reward/feet_silde")
steps = [x.step for x in feet_slide]
values = [x.value for x in feet_slide]
#plt.plot(steps, values, label='Feet slide 1000 envs')

zmp = ea.Scalars("Episode_Reward/zmp_cop_error")
steps = [x.step for x in zmp]
values = [x.value for x in zmp]
#plt.plot(steps, values, label='Recompensa ZMP 1000 envs')

track_xy = ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")
steps = [x.step for x in track_xy]
values = [x.value for x in track_xy]
#plt.plot(steps, values, label='Velocidad lineal para 1000 envs')

track_yaw = ea.Scalars("Episode_Reward/track_ang_vel_z_exp")
steps = [x.step for x in track_yaw]
values = [x.value for x in track_yaw]
plt.plot(steps, values, label='Velocidad angular para 1000 envs')


ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-12_15-56-13")
ea.Reload()
feet_slide = ea.Scalars("Episode_Reward/feet_silde")
steps = [x.step for x in feet_slide]
values = [x.value for x in feet_slide]
#plt.plot(steps, values, label='Feet slide 500 envs')

zmp = ea.Scalars("Episode_Reward/zmp_cop_error")
steps = [x.step for x in zmp]
values = [x.value for x in zmp]
#plt.plot(steps, values, label='Recompensa ZMP 500 envs')

track_xy = ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")
steps = [x.step for x in track_xy]
values = [x.value for x in track_xy]
#plt.plot(steps, values, label='Velocidad lineal para 500 envs')

track_yaw = ea.Scalars("Episode_Reward/track_ang_vel_z_exp")
steps = [x.step for x in track_yaw]
values = [x.value for x in track_yaw]
plt.plot(steps, values, label='Velocidad angular para 500 envs')

#plt.yscale('symlog', linthresh=0.01)
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.legend(fontsize=18)
plt.tight_layout()
plt.title("Recompensa por seguimiento de comando de velocidad angular", fontsize=20)
plt.xlabel("Iteración", fontsize=18)
plt.ylabel("Recompensa", fontsize=18)
plt.grid()
plt.show()


ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-11_20-21-34")
ea.Reload()
plt.figure(figsize=(13, 7))
'''entropy = ea.Scalars("Loss/entropy")
steps = [x.step for x in entropy]
values = [x.value for x in entropy]
#plt.plot(steps, values, label='Loss entropy')

learning_rate = ea.Scalars("Loss/learning_rate")
steps = [x.step for x in learning_rate]
values = [x.value for x in learning_rate]
#plt.plot(steps, values, label='Learning rate')'''

value_function = ea.Scalars("Loss/value_function")
steps = [x.step for x in value_function]
values = [x.value for x in value_function]
plt.plot(steps, values, label='Función de valor de de pérdidia 4000 envs')


surrogate = ea.Scalars("Loss/surrogate")
steps = [x.step for x in surrogate]
values = [x.value for x in surrogate]
#plt.plot(steps, values, label='Objetivo sustituto 4000 envs')

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-12_16-04-34")
ea.Reload()
value_function = ea.Scalars("Loss/value_function")
steps = [x.step for x in value_function]
values = [x.value for x in value_function]
plt.plot(steps, values, label='Función de valor de de pérdidia 1000 envs')


surrogate = ea.Scalars("Loss/surrogate")
steps = [x.step for x in surrogate]
values = [x.value for x in surrogate]
#plt.plot(steps, values, label='Objetivo sustituto 1000 envs')

ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-12_15-56-13")
ea.Reload()
value_function = ea.Scalars("Loss/value_function")
steps = [x.step for x in value_function]
values = [x.value for x in value_function]
plt.plot(steps, values, label='Función de valor de de pérdidia 500 envs')


surrogate = ea.Scalars("Loss/surrogate")
steps = [x.step for x in surrogate]
values = [x.value for x in surrogate]
#plt.plot(steps, values, label='Objetivo sustituto 500 envs')





plt.ylim(0.0, 0.2)
#plt.yscale('symlog', linthresh=0.01)
plt.legend()
plt.title("Funciones de valor de pérdida")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.grid()
plt.show()


ea = event_accumulator.EventAccumulator("logs/rsl_rl/bipedo_manager/2026-04-12_13-25-08")
ea.Reload()


xy_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_xy")
steps = [x.step-1000 for x in xy_vel_error]
values = [x.value for x in xy_vel_error]
values[0:50] = values[50:100]
plt.plot(steps, values, label='Linear velocity reward')

yaw_vel_error = ea.Scalars("Metrics/base_velocity/error_vel_yaw")
steps = [x.step-1000 for x in yaw_vel_error]
values = [x.value for x in yaw_vel_error]
values[0:50] = values[50:100]
plt.plot(steps, values, label='Angular velocity reward')


plt.legend()
plt.title("Command tracking error")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.grid()
plt.show()

feet_slide = ea.Scalars("Episode_Reward/feet_silde")
steps = [x.step-1000 for x in feet_slide]
values = [x.value for x in feet_slide]
plt.plot(steps, values, label='Feet slide')

zmp = ea.Scalars("Episode_Reward/zmp_cop_error")
steps = [x.step-1000 for x in zmp]
values = [x.value for x in zmp]
plt.plot(steps, values, label='ZMP CoP error')

track_xy = ea.Scalars("Episode_Reward/track_lin_vel_xy_exp")
steps = [x.step-1000 for x in track_xy]
values = [x.value for x in track_xy]
plt.plot(steps, values, label='Track xy linear velocity')

track_yaw = ea.Scalars("Episode_Reward/track_ang_vel_z_exp")
steps = [x.step-1000 for x in track_yaw]
values = [x.value for x in track_yaw]
plt.plot(steps, values, label='Track angular velocity')

plt.yscale('symlog', linthresh=0.01)
plt.legend()
plt.title("Main rewards")
plt.xlabel("Iteración")
plt.ylabel("Reward")
plt.grid()
plt.show()