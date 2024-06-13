import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from skimage import io, color
from scipy import signal
from skimage.draw import polygon


class Inpainting:
    def __init__(self, dossier_images, dossier_sauvegardes, dossier_masques):
        self.dossier_images = dossier_images
        self.dossier_sauvegardes = dossier_sauvegardes
        self.dossier_masques = dossier_masques

    def calculer_normales(self, contours):
        normales = []
        for contour in contours:
            normale_i = []
            for i in range(0, len(contour)):
                pt1 = contour[i, 0]
                pt1 = np.array([pt1[0], pt1[1], 0])
                pt2 = contour[(i + 1) % len(contour), 0]
                pt2 = np.array([pt2[0], pt2[1], 0])
                normale = np.cross(pt1 - pt2, np.array([0, 0, 1]))
                normale = normale / np.linalg.norm(normale)
                normale_i.append(normale[0:2].reshape(1, 2))
            normales.append(normale_i)
        return normales

    def calculer_d(self, contours, data, img, taille):
        normales = self.calculer_normales(contours)
        for contour, normale in zip(contours, normales):
            for i in range(len(contour)):
                p = contour[i, 0]
                p = (p[1], p[0])  # openCV indexe en colonnes d'abord
                voisins = self.obtenir_voisins(img, p, taille)
                data[p[0], p[1]] = np.abs(np.dot(self.calculer_isophote(voisins), np.array(normale)[i, 0])) / 255
        return data

    def calculer_c(self, contours, confidence, masque_seuil_0, taille):
        for contour in contours:
            for i in range(len(contour)):
                p = contour[i, 0]
                p = (p[1], p[0])  # openCV indexe en colonnes d'abord
                voisins = self.obtenir_voisins(confidence, p, taille) * self.obtenir_voisins(masque_seuil_0, p, taille) / 255
                confidence[p[0], p[1]] = np.sum(voisins) / taille**2
        return confidence

    def obtenir_voisins(self, img, p, taille):
        voisins = []
        for i in range(taille):
            voisin_ligne = []
            ind_i = p[0] - taille // 2 + i
            if 0 <= ind_i < len(img):
                for j in range(taille):
                    ind_j = p[1] - taille // 2 + j
                    if 0 <= ind_j < len(img[0]):
                        voisin_ligne.append(img[ind_i, ind_j])
                voisins.append(voisin_ligne)
        return np.array(voisins)

    def calculer_isophote(self, voisins):
        grad = np.gradient(voisins)
        gradx, grady = grad[0], grad[1]
        mag = gradx**2 + grady**2
        ind = np.unravel_index(np.argmax(mag, axis=None), mag.shape)
        max_grad = np.array([gradx[ind], grady[ind], 0])
        isophote = np.cross(max_grad, np.array([0, 0, 1]))
        return isophote[0:2] / np.linalg.norm(isophote)

    def trouver_meilleur_point(self, contours, priority):
        meilleure_priorite = np.NINF
        meilleur_p = -1
        for contour in contours:
            for i in range(len(contour)):
                p = contour[i, 0]
                p = (p[1], p[0])  # openCV indexe en colonnes d'abord
                priorite_i = priority[p[0], p[1]]
                if priorite_i > meilleure_priorite:
                    meilleure_priorite = priorite_i
                    meilleur_p = p
        return meilleur_p

    def trouver_meilleur_exemplaire(self, p, img_lab, masque, masque_edge_case, taille):
        r = taille // 2
        masque_voisin = self.obtenir_voisins(np.repeat(masque_edge_case[:, :, np.newaxis], 3, axis=2) / 255, p, taille)
        voisins = self.obtenir_voisins(img_lab, p, taille) * masque_voisin
        meilleure_d = np.inf
        meilleur_exemplaire, meilleure_ind = -1, -1

        pas = max(taille // 9, 1)  # accélérer les grandes images
        for i in range(r + 1, img_lab.shape[0] - r - 2, pas):
            for j in range(r + 1, img_lab.shape[1] - r - 2, pas):
                if np.count_nonzero(masque_edge_case[i - r:i + r + 1, j - r:j + r + 1] == 0) > 0:
                    continue  # ignorer les régions qui incluent la cible
                voisins_i = img_lab[i - r:i + r + 1, j - r:j + r + 1]
                if not voisins.shape == voisins_i.shape:
                    continue
                d = np.linalg.norm((voisins - voisins_i * masque_voisin).flatten())
                if d < meilleure_d:
                    meilleure_d = d
                    meilleur_exemplaire = voisins_i
                    meilleure_ind = np.array([i, j])
        return meilleur_exemplaire, meilleure_ind

    def traiter_image(self, filename):
        print("Lecture de " + filename)
        img = cv2.imread(self.dossier_images + filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = color.rgb2lab(img_rgb)
        masque = cv2.imread(self.dossier_masques + filename.split('.')[0] + '_mask.jpg')
        masque_gris = cv2.cvtColor(masque, cv2.COLOR_BGR2GRAY)
        ret, masque_seuil = cv2.threshold(masque_gris, 50, 255, 0)
        taille = int(min(img.shape[:-1]) / 30 // 2 * 2 + 1)  # La taille de la région de remplissage (arrondie au nombre impair le plus proche)
        print("Taille des texels : ", taille)

        r = taille // 2
        masque_edge_case = np.copy(masque_seuil)  # stocker le masque non modifié
        masque_seuil[:(r + 1), :] = 255
        masque_seuil[-(r + 1):, :] = 255
        masque_seuil[:, :(r + 1)] = 255
        masque_seuil[:, -(r + 1):] = 255

        confidence = masque_seuil / 255
        data = np.zeros(confidence.shape)
        masque_seuil_0 = masque_seuil
        iteration = 0

        while True:
            print("Remplissage de la région " + str(iteration))
            contours, hierarchy = cv2.findContours(masque_seuil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 1:
                break  # condition de sortie si toute la région a été remplie

            contours = contours[1:]  # ignorer le bord de l'image

            confidence = self.calculer_c(contours, confidence, masque_seuil_0, taille)
            data = self.calculer_d(contours, data, img, taille)
            priority = confidence * data

            p = self.trouver_meilleur_point(contours, priority)
            exemplaire, exemplaire_p = self.trouver_meilleur_exemplaire(p, img_lab, masque_seuil_0, masque_edge_case, taille)

            img_lab[p[0] - r:p[0] + r + 1, p[1] - r:p[1] + r + 1] = exemplaire
            img = (color.lab2rgb(img_lab) * 255).astype(np.uint8)

            confidence[p[0] - r:p[0] + r + 1, p[1] - r:p[1] + r + 1] = confidence[p[0], p[1]]
            confidence[:(r + 2), :] = 0
            confidence[-(r + 2):, :] = 0
            confidence[:, :(r + 2)] = 0
            confidence[:, -(r + 2):] = 0

            img_lab = color.rgb2lab(img)
            masque_seuil[p[0] - r:p[0] + r + 1, p[1] - r:p[1] + r + 1] = 255

            if iteration % 20 == 0:
                img_vis = np.array(img)
                cv2.rectangle(img_vis, (p[1] - r, p[0] - r), (p[1] + r + 1, p[0] + r + 1), (255, 0, 0), 1)
                try:
                    cv2.rectangle(img_vis, (exemplaire_p[1] - r, exemplaire_p[0] - r), (exemplaire_p[1] + r + 1, exemplaire_p[0] + r + 1), (0, 0, 255), 1)
                    cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 3)
                    cv2.imwrite(self.dossier_sauvegardes + filename.split('.')[0] + '_confidence.png', (confidence / np.max(confidence) * 255).astype(np.uint8))

                    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.dossier_sauvegardes + filename, np.array(img_bgr))
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde de l'image : {e}")
                    break

            iteration += 1

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.dossier_sauvegardes + filename + "_final", np.array(img_bgr))

    def executer(self):
        for filename in os.listdir(self.dossier_images):
            if not glob.glob(self.dossier_masques + filename.split('.')[0] + '_mask.*'):
                print("L'image " + filename + " n'a pas de masque. Ignorer...")
                continue
            self.traiter_image(filename)


if __name__ == "__main__":
    inpainting = Inpainting(dossier_images='images/', dossier_sauvegardes='output/', dossier_masques='masks/')
    inpainting.executer()
