Mohu中有main和branch两个分支。其中main是最终版的项目文件，删除了无关文件，对代码进行整理完善；branch是备份，冗杂代码较多。
以最终版main的项目为主。
一、文件夹app：
    end_ronghe copy.ipynb是用来检验模型运行效果
    pages文件夹中包含多个可视化网页，通过app/main.py（系统平台主入口）来运行
二、文件夹data：
    初始文件未city_all_new,通过dalete代码来对数据进行处理，最终形成实证分析时使用的数据city_all_7
    sensitivity_coef和sencitivity_metrics是在引入正则化范式时的敏感性分析的数据结果
三、文件夹models：
    建模代码
四、文件夹tests：
    测试代码（性能测试、规模化功能测试）
    测试结果存放在文件夹output文件夹中，包括数据表格、图片
    
