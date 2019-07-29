import pickle

import numpy as np
import argparse
import imageio
import logging
import sys
from scipy.ndimage.filters import convolve

DEFAULT_DISPLACEMENTS_FILE = "final_displacements.pkl"

def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize//2), ksize//2, ksize)
                    ** 2 / 2) / np.sqrt(2*np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""

    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]
    ogMask = mask
    mask = np.append(mask, mask,axis=2)
    mask = np.append(mask, ogMask,axis=2)
    kernelx = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]) / -8.0
    I_x = convolve_img(I, kernelx)
    kernely = kernelx.T
    I_y = convolve_img(I, kernely)
    I_t = I-H

    Ixx = I_x * I_x
    Ixx = Ixx * mask 
    Ixy = I_x * I_y
    Ixy = Ixy * mask 
    Iyy = I_y * I_y
    Iyy = Iyy * mask 
    Ixt = I_x * I_t
    Ixt = Ixt * mask 
    Iyt = I_y * I_t
    Iyt = Iyt * mask 
  
    AtA = np.array([[Ixx.sum(), Ixy.sum()],[Ixy.sum(), Iyy.sum()]])
    Atb = np.array([[Ixt.sum()],[Iyt.sum()]]) * -1.0
    Atb = Atb.flatten()
    displacement = np.linalg.solve(AtA, Atb)
    
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float64)
    for i in range(steps):
        new_H = translate(H, disp)
        temp = lucas_kanade(new_H,I)
        disp += temp[0]

    # Return the final displacement
    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """
    
    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.
    kernel = gaussian_kernel()
    # Add image to the the list as the first level
    pyr = [image]
    for level in range(1, levels):
        # Convolve the previous image with the gussian kernel
        temp = convolve_img(pyr[level - 1], kernel)
        # decimate the convolved image by downsampling the pixels in both dimensions.
        temp = temp[::2,::2]
        
        # add the sampled image to the list
        pyr.append(temp)
    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float16)

    # Build Gaussian pyramids for the two images.
    pyrH = gaussian_pyramid(H,levels)
    pyrI = gaussian_pyramid(I,levels)

    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.

    disp = initial_d / 2.**(levels)
    for level in range(levels):
        # Get the two images for this pyramid level.
        imgH = pyrH[levels - 1 - level]
        imgI = pyrI[levels - 1 - level]

        # Scale the previous level's displacement and apply it to one of the
        # images via translation.
        disp = disp * 2
        imgH = translate(imgH, disp) 

        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        temp = iterative_lucas_kanade(imgH, imgI, steps)

        # Update the displacement based on the one you just computed.
        disp = np.add(disp, temp)

    # Return the final displacement.
    return disp


def build_panorama(images, shape, displacements, initial_position, blend_width=16):
    # Allocate an empty floating-point image with space to store the panorama
    # with the given shape.
    image_height, image_width = images[0].shape[:2]
    pano_height, pano_width = shape
    panorama = np.zeros((pano_height, pano_width, 3), np.float32)

    # Place the last image, warped to align with the first, at its proper place
    # to initialize the panorama.
    cur_pos = initial_position
    cp = np.round(cur_pos).astype(np.int32)
    panorama[cp[0]: cp[0] + image_height, cp[1]: cp[1] +
             image_width] = translate(images[-1], displacements[-1])

    # Place the images at their final positions inside the panorama, blending
    # each image with the panorama in progress. Use a blending window with the
    # given width.
    for i in range(len(images)):
        cp = np.round(cur_pos).astype(np.int32)

        overlap = image_width - abs(displacements[i][0])
        blend_start = int(overlap / 2 - blend_width / 2)
        blend_start_pano = int(cp[1] + blend_start)

        pano_region = panorama[cp[0]: cp[0] + image_height,
                               blend_start_pano: blend_start_pano+blend_width]
        new_region = images[i][:, blend_start: blend_start+blend_width]

        mask = np.zeros((image_height, blend_width, 1), np.float32)
        mask[:] = np.linspace(0, 1, blend_width)[np.newaxis, :, np.newaxis]
        mask[np.all(new_region == 0, axis=2)] = 0
        mask[np.all(pano_region == 0, axis=2)] = 1

        blended_region = mask * new_region + (1-mask) * pano_region

        blended = images[i].copy("C")
        blended[:, blend_start: blend_start+blend_width] = blended_region
        blended[:, :blend_start] = panorama[cp[0] : cp[0] + image_height, cp[1]: blend_start_pano]

        panorama[cp[0]: cp[0] + blended.shape[0],
                 cp[1]: cp[1] + blended.shape[1]] = blended
        cur_pos += -displacements[i][::-1]
        print("Placed %d." % i)

    return panorama

def mosaic(images, initial_displacements, load_displacements_from):
    """Given a list of N images taken in clockwise order and corresponding
    initial X/Y displacements of shape (N,2), refine the displacements and
    build a mosaic.

    initial_displacement[i] gives the translation that should be appiled to
    images[i] to align it with images[(i+1) % N]."""
    N = len(images)

    if load_displacements_from:
        print("Loading saved displacements...")
        final_displacements = pickle.load(open(load_displacements_from, "rb"))
    else:
        print("Refining displacements with Pyramid Iterative Lucas Kanade...")
        final_displacements = []
        for i in range(N):
            disp = pyramid_lucas_kanade(images[i], images[(i+1) % N], initial_displacements[i], 4, 5)
            final_displacements.append(disp)

            # Some debugging output to help diagnose errors.
            print("Image %d:" % i,
                  initial_displacements[i], "->", final_displacements[i], "  ",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -initial_displacements[i]))).mean(), "->",
                  "%0.4f" % abs(
                      (images[i] - translate(images[(i+1) % N], -final_displacements[i]))).mean()
                  )
        print('Saving displacements to ' + DEFAULT_DISPLACEMENTS_FILE)
        pickle.dump(final_displacements, open(DEFAULT_DISPLACEMENTS_FILE, "wb"))


    # Use the final displacements and the images' shape compute the full
    # panorama shape and the starting position for the first panorama image.
    total_x_displacement = 0
    total_y_displacement = 0
    for i in range(len(final_displacements)-1):
        total_x_displacement += abs(final_displacements[i][0])
        total_y_displacement += final_displacements[i][1]

    pano_width = int(np.ceil(images[0].shape[1] + total_x_displacement))
    pano_height = int(np.ceil(images[0].shape[0] + total_y_displacement))
    initial_pos = [total_y_displacement, 0]

    # Build the panorama.
    print("Building panorama...")
    panorama = build_panorama(images, (pano_height, pano_width), final_displacements, initial_pos.copy())
    return panorama, final_displacements


def warp_panorama(images, panorama, final_displacements):

    warped = panorama
    return warped


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Creates a mosaic by stitching together a provided set of images.')
    parser.add_argument(
        'input', type=str, help='A txt file containing the images and initial displacement positions.')
    parser.add_argument('output', type=str,
                        help='What image file to save the panorama to.')
    parser.add_argument('--displacements', type=str,
                        help='Load displacements from this pickle file (useful for build_panorama).', default=None)
    args = parser.parse_args()

    filenames, xinit, yinit = zip(
        *[l.strip().split() for l in open(args.input).readlines()])
    xinit = np.array([float(x) for x in xinit])[:, np.newaxis]
    yinit = np.array([float(y) for y in yinit])[:, np.newaxis]
    disps = np.hstack([xinit, yinit])

    images = [imageio.imread(fn)[:, :, :3].astype(
        np.float32)/255. for fn in filenames]

    panorama, final_displacements = mosaic(images, disps, args.displacements)

    result = warp_panorama(images, panorama, final_displacements)
    imageio.imwrite(args.output, result)
