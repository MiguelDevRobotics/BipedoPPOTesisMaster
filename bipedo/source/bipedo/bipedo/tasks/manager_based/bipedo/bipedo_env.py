'''Copyright 2026 Miguel de Jesus Santiago Vasquez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'''

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import torch
from collections.abc import Sequence
from typing import Any


class BipedoEnv(ManagerBasedRLEnv):

    """
    Entorno personalizado que hereda de ManagerBasedRLEnv para agregar 
    el seguimiento de la métrica de 'distancia recorrida por episodio'.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # 1. Inicializar la clase base primero
        super().__init__(cfg, render_mode, **kwargs)

        # 2. Inicializar los tensores para el tracking de distancias
        # Usamos self.num_envs y self.device, que son configurados por la clase base
        self.distance_traveled = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_positions = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # definido en el SceneManager (cfg.scene).
        self.robot_entity = "robot" 
        
        print("[INFO]: Custom Environment initialized with distance tracking.")

    def step(self, action: torch.Tensor):
        # Acceso directo (más rápido)
        self.previous_positions[:] = self.scene[self.robot_entity].data.root_pos_w
        
        obs, rewards, terminated, time_outs, extras = super().step(action)

        # Cálculo directo
        current_positions = self.scene[self.robot_entity].data.root_pos_w
        step_distance = torch.norm(current_positions - self.previous_positions, dim=-1)
        
        step_distance[self.reset_buf] = 0.0
        self.distance_traveled += step_distance

        return obs, rewards, terminated, time_outs, extras
    
    def _reset_idx(self, env_ids: Sequence[int]):
        # 1. Calculamos la media de la distancia para los envs que terminan
        # Usamos .item() para asegurarnos de que sea un float puro de Python
        mean_distance = 0.0
        if len(env_ids) > 0:
            mean_distance = self.distance_traveled[env_ids].mean().item()

        # 2. Llamar al reset de la clase base (esto inicializa self.extras["log"])
        super()._reset_idx(env_ids)

        # 3. AGREGAR LA MÉTRICA DE FORMA PLANA
        if len(env_ids) > 0:
            # Al ponerlo directamente aquí, el wrapper lo procesará correctamente
            self.extras["log"]["Metrics/distancia_recorrida"] = mean_distance

        # 4. Reiniciar el contador local
        self.distance_traveled[env_ids] = 0.0
