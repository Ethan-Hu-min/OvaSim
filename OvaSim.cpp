//#include "Renderer.h"
//#include "GLFWindow.h"
//#include <gl/GL.h>
//
//struct SampleWindow : public GLFCameraWindow {
//    SampleWindow(const std::string& title,
//        const Scene* scene,
//        const Camera& camera,
//        const float worldScale)
//        : GLFCameraWindow(title, camera.from, camera.at, camera.up, worldScale),
//        sample(scene)
//    {
//        sample.setCamera(camera);
//    }
//
//    virtual void render() override
//    {
//        if (cameraFrame.modified) {
//            sample.setCamera(Camera{ cameraFrame.get_from(),
//                                     cameraFrame.get_at(),
//                                     cameraFrame.get_up() });
//            cameraFrame.modified = false;
//        }
//        //sample.getCamera();
//        sample.render();
//    }
//
//    virtual void draw() override
//    {
//
//        //std::cout << "start draw" << std::endl;
//        sample.downloadPixels(pixels.data());
//        //std::cout << fbSize.x << " " << fbSize.y<< " " << pixels[0] << " " <<pixels.size() << std::endl;
//        //std::cout << fbTexture << std::endl;
//        gladLoadGL();
//        if (fbTexture == 0)
//            glGenTextures(1, &fbTexture);
//        glBindTexture(GL_TEXTURE_2D, fbTexture);
//        GLenum texFormat = GL_RGBA;
//        GLenum texelType = GL_UNSIGNED_BYTE;
//        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
//            texelType, pixels.data());
//
//        glDisable(GL_LIGHTING);
//        glColor3f(1, 1, 1);
//
//        glMatrixMode(GL_MODELVIEW);
//        glLoadIdentity();
//
//        glEnable(GL_TEXTURE_2D);
//        glBindTexture(GL_TEXTURE_2D, fbTexture);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//
//        glDisable(GL_DEPTH_TEST);
//
//        glViewport(0, 0, fbSize.x, fbSize.y);
//
//        glMatrixMode(GL_PROJECTION);
//        glLoadIdentity();
//        glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);
//
//        glBegin(GL_QUADS);
//        {
//            glTexCoord2f(0.f, 0.f);
//            glVertex3f(0.f, 0.f, 0.f);
//
//            glTexCoord2f(0.f, 1.f);
//            glVertex3f(0.f, (float)fbSize.y, 0.f);
//
//            glTexCoord2f(1.f, 1.f);
//            glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);
//
//            glTexCoord2f(1.f, 0.f);
//            glVertex3f((float)fbSize.x, 0.f, 0.f);
//        }
//        glEnd();
//    }
//
//    virtual void resize(const vec2i& newSize)
//    {
//        fbSize = newSize;
//        sample.resize(newSize);
//        pixels.resize(newSize.x * newSize.y);
//    }
//
//    vec2i                 fbSize;
//    GLuint                fbTexture{ 0 };
//    Renderer        sample;
//    std::vector<uint32_t> pixels;
//
//};
//
//
//int main() {
//	std::string exampleDir = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK/OvaSim/data/example1/";
//	std::string sceneDir = "example1.scene";
//	Scene* scene = new Scene();
//    //std::unique_ptr<Scene> scene = std::make_unique<Scene>();
//	scene->setSceneDir(exampleDir);
//	std::cout << "start" << std::endl;
//	scene->parseConfig(sceneDir);
//	std::cout << "finish" << std::endl;
//	//std::cout << scene.models.size() << std::endl;
//	scene->loadModels();
//	scene->createWorldModels();
//
//	std::cout << scene->models[0].meshes[0]->index[0] << std::endl;
//	std::cout << scene->models[0].meshes[0]->index[1] << std::endl;
//	std::cout << scene->models[0].meshes[0]->index[2] << std::endl;
//	for (int i = 0; i < scene->models.size(); i++) {
//		std::cout << scene->models[i].meshes.size() << std::endl;
//	}
//	std::cout << scene->worldmodel.size();
//    try {
//        Camera camera = { vec3f(8.0f, 20.0f, -25.0f), vec3f(-0.221381f, -0.633619f, 0.741294f), vec3f(-0.119666f,0.510606f,0.400702f) };
//        const float worldScale = 20.0f;
//        SampleWindow* window = new SampleWindow("MY Renderer", scene, camera, worldScale);
//        window->run();
//    }
//    catch(std::runtime_error& e){
//        std::cout << GDT_TERMINAL_RED << "FATAL ERROR: " << e.what()
//            << GDT_TERMINAL_DEFAULT << std::endl;
//        exit(1);
//    }
//
//	return 0;
//}



#include "USRenderer.h"
int main() {
	std::string exampleDir = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK/OvaSim/data/example1/";
	std::string sceneDir = "example1.scene";
	Scene* scene = new Scene();
    //std::unique_ptr<Scene> scene = std::make_unique<Scene>();
	scene->setSceneDir(exampleDir);
	std::cout << "start" << std::endl;
	scene->parseConfig(sceneDir);
	std::cout << "finish" << std::endl;
	//std::cout << scene.models.size() << std::endl;
	scene->loadModels();
	scene->createWorldModels();
	for (int i = 0; i < scene->models.size(); i++) {
		std::cout << scene->models[i].meshes.size() << std::endl;
	}
	std::cout << scene->worldmodel.size();
	scene->setTransducer(vec3f(-50.0f, -100.0f, 150.0f),
		vec3f(0.197562f, 0.729760f, -0.654538f), vec3f(0.728110f, -0.556300f, -0.400482f), 512, 90.0,512.0, 512.0);

	USRenderer* render = new USRenderer(scene);
	render->setTransducer(scene->transducer);
	render->run();
	return 0;
}

