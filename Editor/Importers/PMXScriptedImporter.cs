using UMT;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace UMT.Editor
{
    /// <summary>
    /// Unity <see cref="ScriptedImporter"/> for <c>.pmx</c> assets that runs the full PMX import pipeline and, optionally, converts attached VMD animations into <see cref="AnimationClip"/> sub-assets.
    /// </summary>
    [ScriptedImporter(1, new[] { "pmx" })]
    public sealed class PMXScriptedImporter : ScriptedImporter
    {
        [SerializeField] private bool m_CreateAvatar = false;
        [SerializeField] private bool m_GenerateDebugData = false;
        [SerializeField] private PMXMaterialCreationMode m_MaterialCreationMode = PMXMaterialCreationMode.Standard;
        [SerializeField] private List<VMDAnimation> m_VMDAnimations = new List<VMDAnimation>();
        [SerializeField] private float m_VMDFrameRate = 30.0f;
        [SerializeField] private bool m_VMDBakeIKToFK = true;
        [SerializeField] private bool m_VMDBakePhysicsToFK = false;
        [SerializeField] private float m_VMDPhysicsWarmUpDuration = 5.0f;

        /// <summary>
        /// Declares that generated <see cref="Material"/> sub-assets can be remapped to external assets, enabling the Materials tab's per-slot overrides and "Extract Materials..." action.
        /// </summary>
        /// <param name="type">Sub-asset type being queried for remap support.</param>
        /// <returns>True for <see cref="Material"/>; otherwise false.</returns>
        public override bool SupportsRemappedAssetType(System.Type type)
        {
            return type == typeof(Material);
        }

        /// <summary>
        /// Imports the <c>.pmx</c> asset: builds the model, materials, meshes, MMD runtime components, and optional avatar, registers them with the import context, and converts any attached VMD animations to clips.
        /// </summary>
        /// <param name="ctx">Asset import context for the source PMX file.</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            UMTTimingCollector timingCollector = new UMTTimingCollector();
            PMXImportOptions options = new PMXImportOptions
            {
                umtResources = PMXImportPipeline.LoadRequiredResources(),
                createAvatar = m_CreateAvatar,
                applyRenames = true,
                strictVersion = true,
                parent = null,
                loadTextures = (model, opts) => PMXProjectTextures.Load(model, ctx.assetPath),
                timingCallback = timingCollector.RecordTiming,
                modelPostprocess = PMXImportHooks.For(ctx.assetPath),
            };

            if (m_MaterialCreationMode == PMXMaterialCreationMode.Override)
            {
                Dictionary<string, Material> materialOverrides = new Dictionary<string, Material>();
                foreach (KeyValuePair<AssetImporter.SourceAssetIdentifier, Object> remap in GetExternalObjectMap())
                {
                    if (remap.Key.type == typeof(Material) && remap.Value is Material overrideMaterial && overrideMaterial != null)
                    {
                        materialOverrides[remap.Key.name] = overrideMaterial;
                    }
                }
                options.materialOverrides = materialOverrides;
            }

            PMXImportResult importResult = PMXImporter.Import(ctx.assetPath, options);
            PMXImportPipelineResult pipelineResult = PMXImportPipeline.Import(importResult, ctx.assetPath, ctx, m_GenerateDebugData);

            PMXModel model = importResult.model;
            if (model != null && m_VMDAnimations.Count > 0)
            {
                VMDAnimationClipOptions vmdOptions = new VMDAnimationClipOptions
                {
                    frameRate = Mathf.Max(1.0f, m_VMDFrameRate),
                    bakeIKToFK = m_VMDBakeIKToFK,
                    bakePhysicsToFK = m_VMDBakePhysicsToFK,
                    physicsSeed = 0,
                    physicsWarmUpDuration = Mathf.Max(0.0f, m_VMDPhysicsWarmUpDuration),
                    timingCallback = timingCollector.RecordTiming,
                };

                for (int i = 0; i < m_VMDAnimations.Count; ++i)
                {
                    VMDAnimation vmdAnimation = m_VMDAnimations[i];
                    if (vmdAnimation == null)
                    {
                        continue;
                    }

                    string clipName = string.IsNullOrEmpty(vmdAnimation.name) ? "VMDClip" : $"{vmdAnimation.name}_Clip";
                    string progressTitle = $"Converting VMD {i + 1}/{m_VMDAnimations.Count}";
                    VMDModelClipData clipData = VMDAnimationClipConverter.Convert(vmdAnimation, model, vmdOptions, (stage, frame, totalFrames) => VMDClipProgress.Report(progressTitle, clipName, stage, frame, totalFrames));
                    AnimationClip clip = VMDClipDataBuilder.BuildAnimationClip(clipData, vmdOptions.frameRate);
                    clip.name = clipName;
                    ctx.AddObjectToAsset(clipName, clip);
                }

                EditorUtility.ClearProgressBar();
            }

            string timingReport = timingCollector.BuildReport("PMX Import Total");
            if (m_GenerateDebugData)
            {
                string timingLogPath = Path.ChangeExtension(ctx.assetPath, ".pmx.import.log");
                File.WriteAllText(timingLogPath, timingReport);
            }

            Debug.Log($"[PMX Import] {ctx.assetPath}: model={pipelineResult.modelName}, " + $"meshes={pipelineResult.meshCount}, materials={pipelineResult.materialCount}, " + $"remapped={pipelineResult.remappedMaterialCount}, " + $"avatar={pipelineResult.hasAvatar}, renamed={pipelineResult.renameCount}, " + $"vmdClips={m_VMDAnimations.Count}\n" + timingReport);
        }
    }

    /// <summary>
    /// Project-side hooks into <see cref="PMXScriptedImporter"/>. Nothing is registered by default, in which case imports are unchanged.
    /// </summary>
    public static class PMXImportHooks
    {
        /// <summary>
        /// Invoked with the freshly parsed model and the asset path of the <c>.pmx</c> being imported, before renames and before any Unity object is built (<see cref="PMXImportOptions.modelPostprocess"/>).
        /// Handlers may edit the model in place. An exception thrown by a handler fails the import and skips the remaining handlers, so handlers should catch their own errors. Register from an <c>[InitializeOnLoad]</c> class. Unity does not know what a handler reads, so the project must reimport the affected assets itself when its inputs change.
        /// </summary>
        public static event System.Action<PMXModel, string> modelRead;

        internal static System.Action<PMXModel> For(string assetPath)
        {
            System.Action<PMXModel, string> handlers = modelRead;
            if (handlers == null)
            {
                return null;
            }
            return model => handlers(model, assetPath);
        }
    }
}
