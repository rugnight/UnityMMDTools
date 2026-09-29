using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

namespace UMT
{
    public static partial class VMDAnimationClipConverter
    {
        // The 7 non-baked bone channels, in the order stored in VMDClipData.curves[7 * boneIndex + channel]: localPosition x/y/z, localRotation x/y/z/w.
        // R02 fork: non-baked bones are written as dense per-frame curves interpolated with the VMD Bezier/slerp (the same interpolation the baked
        // path uses), without baking IK. The former sparse localEulerAnglesRaw curves let Unity interpolate Euler angles between keys, which does not
        // follow VMD's quaternion slerp: bones took a different path between keys (measured up to 178 degrees off on a pole-dance motion).
        private const int k_NonBakedBoneChannelCount = 7;

        private static void AddBoneCurves(VMDClipData bones, VMDAnimation animation, PMXModel model, ref MMDTransformManager.SolverContext transformContext, string[] bonePaths, ref IndexResolver resolver, float frameRate, ProgressCallback progress)
        {
            uint lastFrame = GetLastBoneFrame(animation);
            int frameCount = checked((int)lastFrame + 1);
            ReportProgress(progress, Stage.BoneConversion, 0, frameCount);

            ref NativeArray<MMDBoneTransform.BoneSolverConfig> boneConfigData = ref transformContext.boneConfigData;
            int boneCount = boneConfigData.Length;

            NativeList<ResolvedBoneFrame> resolvedBoneFrames = BuildSortedResolvedBoneFrames(animation, ref resolver, in boneConfigData, boneCount, frameCount, Allocator.Persistent);
            NativeArray<bool> sourceBoneSelection = new NativeArray<bool>(boneCount, Allocator.Persistent);
            NativeArray<int> sourceTrackIndexByBone = new NativeArray<int>(boneCount, Allocator.Persistent);
            NativeArray<int> lastFrameWithSampleByBone = new NativeArray<int>(boneCount, Allocator.Persistent);
            NativeList<int> sourceBoneIndices = new NativeList<int>(Allocator.Persistent);
            AnimationMath.BuildSourceBoneTracks(in resolvedBoneFrames, ref sourceBoneSelection, ref sourceTrackIndexByBone, ref sourceBoneIndices);

            NativeArray<BoneSample> boneSamples = new NativeArray<BoneSample>(checked(sourceBoneIndices.Length * frameCount), Allocator.Persistent);
            AnimationMath.FillCompactBoneSamples(in resolvedBoneFrames, in sourceTrackIndexByBone, ref boneSamples, frameCount);
            AnimationMath.InterpolateCompactBoneSamples(in boneConfigData, in sourceBoneIndices, ref boneSamples, frameCount, ref lastFrameWithSampleByBone);

            for (int trackIndex = 0; trackIndex < sourceBoneIndices.Length; ++trackIndex)
            {
                int boneIndex = sourceBoneIndices[trackIndex];
                bool writePosition = CanWritePositionCurves(model, boneIndex);
                bool writeRotation = CanWriteRotationCurves(model, boneIndex);
                if (!writePosition && !writeRotation)
                {
                    continue;
                }

                int sampleStart = trackIndex * frameCount;
                int keyCount = lastFrameWithSampleByBone[boneIndex] + 1;
                Keyframe[] positionX = new Keyframe[keyCount];
                Keyframe[] positionY = new Keyframe[keyCount];
                Keyframe[] positionZ = new Keyframe[keyCount];
                Keyframe[] rotationX = new Keyframe[keyCount];
                Keyframe[] rotationY = new Keyframe[keyCount];
                Keyframe[] rotationZ = new Keyframe[keyCount];
                Keyframe[] rotationW = new Keyframe[keyCount];
                quaternion previousRotation = quaternion.identity;
                for (int frameIndex = 0; frameIndex < keyCount; ++frameIndex)
                {
                    BoneSample sample = boneSamples[sampleStart + frameIndex];
                    float time = frameIndex / frameRate;
                    quaternion rotation = sample.rotation;
                    // Keep neighbouring keys in the same hemisphere so Unity's per-component interpolation does not take the long way round.
                    if (frameIndex > 0 && math.dot(previousRotation, rotation) < 0.0f)
                    {
                        rotation = new quaternion(-rotation.value);
                    }
                    previousRotation = rotation;

                    positionX[frameIndex] = new Keyframe(time, sample.position.x);
                    positionY[frameIndex] = new Keyframe(time, sample.position.y);
                    positionZ[frameIndex] = new Keyframe(time, sample.position.z);
                    rotationX[frameIndex] = new Keyframe(time, rotation.value.x);
                    rotationY[frameIndex] = new Keyframe(time, rotation.value.y);
                    rotationZ[frameIndex] = new Keyframe(time, rotation.value.z);
                    rotationW[frameIndex] = new Keyframe(time, rotation.value.w);
                }

                int channelStart = checked(boneIndex * k_NonBakedBoneChannelCount);
                bones.paths[boneIndex] = bonePaths[boneIndex];
                if (writePosition)
                {
                    bones.curves[channelStart + 0] = BuildDenseCurve(positionX);
                    bones.curves[channelStart + 1] = BuildDenseCurve(positionY);
                    bones.curves[channelStart + 2] = BuildDenseCurve(positionZ);
                }

                if (writeRotation)
                {
                    bones.curves[channelStart + 3] = BuildDenseCurve(rotationX);
                    bones.curves[channelStart + 4] = BuildDenseCurve(rotationY);
                    bones.curves[channelStart + 5] = BuildDenseCurve(rotationZ);
                    bones.curves[channelStart + 6] = BuildDenseCurve(rotationW);
                }
            }

            boneSamples.Dispose();
            sourceBoneIndices.Dispose();
            lastFrameWithSampleByBone.Dispose();
            sourceTrackIndexByBone.Dispose();
            sourceBoneSelection.Dispose();
            resolvedBoneFrames.Dispose();

            ReportProgress(progress, Stage.BoneConversion, frameCount, frameCount);
        }

        private static AnimationCurve BuildDenseCurve(Keyframe[] keyframes)
        {
            ApplyLinearTangents(keyframes);
            return new AnimationCurve(keyframes);
        }
    }
}
